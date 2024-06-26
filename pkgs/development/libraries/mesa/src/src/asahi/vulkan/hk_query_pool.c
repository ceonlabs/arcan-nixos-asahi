/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_query_pool.h"

#include "agx_compile.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_event.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "shader_enums.h"
#include "vk_common_entrypoints.h"
#include "vk_meta.h"
#include "vk_pipeline.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/libagx_shaders.h"
#include "asahi/lib/shaders/query.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

#include "util/os_time.h"
#include "vulkan/vulkan_core.h"

struct hk_query_report {
   uint64_t value;
   uint64_t timestamp;
};

static uint16_t *
hk_pool_oq_index_ptr(const struct hk_query_pool *pool)
{
   return (uint16_t *)(pool->bo->ptr.cpu + pool->query_start);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_query_pool *pool;

   bool occlusion = pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION;
   unsigned occlusion_queries = occlusion ? pCreateInfo->queryCount : 0;

   pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* We place the availability first and then data */
   pool->query_start = align(pool->vk.query_count * sizeof(uint32_t),
                             sizeof(struct hk_query_report));

   uint32_t reports_per_query;
   switch (pCreateInfo->queryType) {
   case VK_QUERY_TYPE_OCCLUSION:
      /* Specially handled as part of the occlusion heap */
      reports_per_query = 0;
      break;
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
      reports_per_query = 2;
      break;
   case VK_QUERY_TYPE_TIMESTAMP:
      reports_per_query = 1;
      break;
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      reports_per_query = 2 * util_bitcount(pool->vk.pipeline_statistics);
      break;
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      // 2 for primitives succeeded 2 for primitives needed
      reports_per_query = 4;
      break;
   default:
      unreachable("Unsupported query type");
   }
   pool->query_stride = reports_per_query * sizeof(struct hk_query_report);

   if (pool->vk.query_count > 0) {
      uint32_t bo_size = pool->query_start;

      /* For occlusion queries, we stick the query index remapping here */
      if (occlusion_queries)
         bo_size += sizeof(uint16_t) * pool->vk.query_count;
      else
         bo_size += pool->query_stride * pool->vk.query_count;

      pool->bo =
         agx_bo_create(&dev->dev, bo_size, AGX_BO_WRITEBACK, "Query pool");
      if (!pool->bo) {
         hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   uint16_t *oq_index = hk_pool_oq_index_ptr(pool);

   for (unsigned i = 0; i < occlusion_queries; ++i) {
      uint64_t zero = 0;
      unsigned index;

      VkResult result = hk_descriptor_table_add(
         dev, &dev->occlusion_queries, &zero, sizeof(uint64_t), &index);

      if (result != VK_SUCCESS) {
         hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      /* We increment as we go so we can clean up properly if we run out */
      assert(pool->oq_queries < occlusion_queries);
      oq_index[pool->oq_queries++] = index;
   }

   *pQueryPool = hk_query_pool_to_handle(pool);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyQueryPool(VkDevice device, VkQueryPool queryPool,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   if (!pool)
      return;

   uint16_t *oq_index = hk_pool_oq_index_ptr(pool);

   for (unsigned i = 0; i < pool->oq_queries; ++i) {
      hk_descriptor_table_remove(dev, &dev->occlusion_queries, oq_index[i]);
   }

   agx_bo_unreference(pool->bo);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

static uint64_t
hk_query_available_addr(struct hk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return pool->bo->ptr.gpu + query * sizeof(uint32_t);
}

static nir_def *
hk_nir_available_addr(nir_builder *b, nir_def *pool_addr, nir_def *query)
{
   nir_def *offset = nir_imul_imm(b, query, sizeof(uint32_t));
   return nir_iadd(b, pool_addr, nir_u2u64(b, offset));
}

static uint32_t *
hk_query_available_map(struct hk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return (uint32_t *)pool->bo->ptr.cpu + query;
}

static uint64_t
hk_query_offset(struct hk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return pool->query_start + query * pool->query_stride;
}

static uint64_t
hk_query_report_addr(struct hk_device *dev, struct hk_query_pool *pool,
                     uint32_t query)
{

   if (pool->oq_queries) {
      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);
      return dev->occlusion_queries.bo->ptr.gpu +
             (oq_index[query] * sizeof(uint64_t));
   } else {
      return pool->bo->ptr.gpu + hk_query_offset(pool, query);
   }
}

static nir_def *
hk_nir_query_report_addr(nir_builder *b, nir_def *pool_addr,
                         nir_def *query_start, nir_def *query_stride,
                         nir_def *query)
{
   nir_def *offset =
      nir_iadd(b, query_start, nir_umul_2x32_64(b, query, query_stride));
   return nir_iadd(b, pool_addr, offset);
}

static struct hk_query_report *
hk_query_report_map(struct hk_device *dev, struct hk_query_pool *pool,
                    uint32_t query)
{
   if (pool->oq_queries) {
      uint64_t *queries = (uint64_t *)dev->occlusion_queries.bo->ptr.cpu;
      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);

      return (struct hk_query_report *)&queries[oq_index[query]];
   } else {
      return (void *)((char *)pool->bo->ptr.cpu + hk_query_offset(pool, query));
   }
}

struct hk_write_params {
   uint64_t address;
   uint32_t value;
};

VkResult
hk_build_write_shader(struct hk_device *dev, struct hk_shader **s)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "write");

   nir_def *addr = nir_load_preamble(
      &b, 1, 64, .base = offsetof(struct hk_write_params, address) / 2);

   nir_def *value = nir_load_preamble(
      &b, 1, 32, .base = offsetof(struct hk_write_params, value) / 2);

   nir_store_global(&b, addr, 4, value, nir_component_mask(1));

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   return hk_compile_shader(dev, &info, NULL, NULL, s);
}

void
hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value)
{
   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.post_gfx, true);
   if (!cs)
      return;

   /* XXX: todo: stream link would be faster */
   if (cs->current + 0x2000 > cs->end) {
      hk_cmd_buffer_end_compute_internal(&cmd->current_cs.post_gfx);
      cs = hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.post_gfx, true);
      if (!cs)
         return;
   }

   /* As soon as we mark a query available, it needs to be available system
    * wide, otherwise a CPU-side get result can query. As such, we cache flush
    * before and then let coherency works its magic. Without this barrier, we
    * get flakes in
    *
    * dEQP-VK.query_pool.occlusion_query.get_results_conservative_size_64_wait_query_without_availability_draw_triangles_discard
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   hk_cdm_cache_flush(dev, cs);

   struct hk_shader *s = dev->write_shader;
   struct hk_write_params params = {.address = address, .value = value};
   uint32_t usc = hk_upload_usc_words_kernel(cmd, s, &params, sizeof(params));

   hk_dispatch(dev, cs, s, usc, 1, 1, 1);
}

/**
 * Goes through a series of consecutive query indices in the given pool,
 * setting all element values to 0 and emitting them as available.
 */
static void
emit_zero_queries(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   for (uint32_t i = 0; i < num_queries; i++) {
      hk_queue_write(cmd, hk_query_available_addr(pool, first_index + i), 1);
      hk_queue_write(cmd, hk_query_report_addr(dev, pool, first_index + i), 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_ResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                  uint32_t queryCount)
{
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   uint32_t *available = hk_query_available_map(pool, firstQuery);
   memset(available, 0, queryCount * sizeof(*available));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                     uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* XXX: perf */
   for (uint32_t i = 0; i < queryCount; i++) {
      uint64_t addr = hk_query_available_addr(pool, firstQuery + i);
      hk_queue_write(cmd, addr, 0);

      /* XXX: is this supposed to happen on the begin? */
      hk_queue_write(cmd, hk_query_report_addr(dev, pool, firstQuery + i), 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                      VkPipelineStageFlags2 stage, VkQueryPool queryPool,
                      uint32_t query)
{
   unreachable("todo");
#if 0
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   struct nv_push *p = hk_cmd_buffer_push(cmd, 10);

   uint64_t report_addr = hk_query_report_addr(pool, query);
   P_MTHD(p, NV9097, SET_REPORT_SEMAPHORE_A);
   P_NV9097_SET_REPORT_SEMAPHORE_A(p, report_addr >> 32);
   P_NV9097_SET_REPORT_SEMAPHORE_B(p, report_addr);
   P_NV9097_SET_REPORT_SEMAPHORE_C(p, 0);
   P_NV9097_SET_REPORT_SEMAPHORE_D(p, {
      .operation = OPERATION_REPORT_ONLY,
      .pipeline_location = vk_stage_flags_to_nv9097_pipeline_location(stage),
      .structure_size = STRUCTURE_SIZE_FOUR_WORDS,
   });

   uint64_t available_addr = hk_query_available_addr(pool, query);
   P_MTHD(p, NV9097, SET_REPORT_SEMAPHORE_A);
   P_NV9097_SET_REPORT_SEMAPHORE_A(p, available_addr >> 32);
   P_NV9097_SET_REPORT_SEMAPHORE_B(p, available_addr);
   P_NV9097_SET_REPORT_SEMAPHORE_C(p, 1);
   P_NV9097_SET_REPORT_SEMAPHORE_D(p, {
      .operation = OPERATION_RELEASE,
      .release = RELEASE_AFTER_ALL_PRECEEDING_WRITES_COMPLETE,
      .pipeline_location = PIPELINE_LOCATION_ALL,
      .structure_size = STRUCTURE_SIZE_ONE_WORD,
   });

   /* From the Vulkan spec:
    *
    *   "If vkCmdWriteTimestamp2 is called while executing a render pass
    *    instance that has multiview enabled, the timestamp uses N consecutive
    *    query indices in the query pool (starting at query) where N is the
    *    number of bits set in the view mask of the subpass the command is
    *    executed in. The resulting query values are determined by an
    *    implementation-dependent choice of one of the following behaviors:"
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries, as described in the first behavior listed in the
    * Vulkan spec:
    *
    *   "The first query is a timestamp value and (if more than one bit is set
    *   in the view mask) zero is written to the remaining queries."
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1);
   }
#endif
}

static void
hk_cmd_begin_end_query(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                       uint32_t query, uint32_t index,
                       VkQueryControlFlags flags, bool end)
{
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      assert(query < pool->oq_queries);

      if (end) {
         cmd->state.gfx.occlusion.mode = AGX_VISIBILITY_MODE_NONE;

         /* We need to set available=1 after the graphics work finishes. */
         hk_queue_write(cmd, hk_query_available_addr(pool, query), 1);
      } else {
         cmd->state.gfx.occlusion.mode = flags & VK_QUERY_CONTROL_PRECISE_BIT
                                            ? AGX_VISIBILITY_MODE_COUNTING
                                            : AGX_VISIBILITY_MODE_BOOLEAN;
      }

      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);
      cmd->state.gfx.occlusion.index = oq_index[query];
      cmd->state.gfx.dirty |= HK_DIRTY_OCCLUSION;
      break;
   }

   default:
      unreachable("Unsupported query type");
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t query, VkQueryControlFlags flags,
                           uint32_t index)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   hk_cmd_begin_end_query(cmd, pool, query, index, flags, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                         uint32_t query, uint32_t index)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   hk_cmd_begin_end_query(cmd, pool, query, index, 0, true);

   /* From the Vulkan spec:
    *
    *   "If queries are used while executing a render pass instance that has
    *    multiview enabled, the query uses N consecutive query indices in
    *    the query pool (starting at query) where N is the number of bits set
    *    in the view mask in the subpass the query is used in. How the
    *    numerical results of the query are distributed among the queries is
    *    implementation-dependent."
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries.
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1);
   }
}

static bool
hk_query_is_available(struct hk_query_pool *pool, uint32_t query)
{
   uint32_t *available = hk_query_available_map(pool, query);
   return p_atomic_read(available) != 0;
}

#define HK_QUERY_TIMEOUT 2000000000ull

static VkResult
hk_query_wait_for_available(struct hk_device *dev, struct hk_query_pool *pool,
                            uint32_t query)
{
   uint64_t abs_timeout_ns = os_time_get_absolute_timeout(HK_QUERY_TIMEOUT);

   while (os_time_get_nano() < abs_timeout_ns) {
      if (hk_query_is_available(pool, query))
         return VK_SUCCESS;

      VkResult status = vk_device_check_status(&dev->vk);
      if (status != VK_SUCCESS)
         return status;
   }

   return vk_device_set_lost(&dev->vk, "query timeout");
}

static void
cpu_write_query_result(void *dst, uint32_t idx, VkQueryResultFlags flags,
                       uint64_t result)
{
   if (flags & VK_QUERY_RESULT_64_BIT) {
      uint64_t *dst64 = dst;
      dst64[idx] = result;
   } else {
      uint32_t *dst32 = dst;
      dst32[idx] = result;
   }
}

static void
cpu_get_query_delta(void *dst, const struct hk_query_report *src, uint32_t idx,
                    VkQueryResultFlags flags)
{
   uint64_t delta = src[idx * 2 + 1].value - src[idx * 2].value;
   cpu_write_query_result(dst, idx, flags, delta);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetQueryPoolResults(VkDevice device, VkQueryPool queryPool,
                       uint32_t firstQuery, uint32_t queryCount,
                       size_t dataSize, void *pData, VkDeviceSize stride,
                       VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   VkResult status = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t query = firstQuery + i;

      bool available = hk_query_is_available(pool, query);

      if (!available && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         status = hk_query_wait_for_available(dev, pool, query);
         if (status != VK_SUCCESS)
            return status;

         available = true;
      }

      bool write_results = available || (flags & VK_QUERY_RESULT_PARTIAL_BIT);

      const struct hk_query_report *src = hk_query_report_map(dev, pool, query);
      assert(i * stride < dataSize);
      void *dst = (char *)pData + i * stride;

      uint32_t available_dst_idx = 1;
      switch (pool->vk.query_type) {
      case VK_QUERY_TYPE_OCCLUSION:
         if (write_results)
            cpu_write_query_result(dst, 0, flags, src[0].value);
         break;

      case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
         if (write_results)
            cpu_get_query_delta(dst, src, 0, flags);
         break;
      case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
         uint32_t stat_count = util_bitcount(pool->vk.pipeline_statistics);
         available_dst_idx = stat_count;
         if (write_results) {
            for (uint32_t j = 0; j < stat_count; j++)
               cpu_get_query_delta(dst, src, j, flags);
         }
         break;
      }
      case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT: {
         const int prims_succeeded_idx = 0;
         const int prims_needed_idx = 1;
         available_dst_idx = 2;
         if (write_results) {
            cpu_get_query_delta(dst, src, prims_succeeded_idx, flags);
            cpu_get_query_delta(dst, src, prims_needed_idx, flags);
         }
         break;
      }
      case VK_QUERY_TYPE_TIMESTAMP:
         if (write_results)
            cpu_write_query_result(dst, 0, flags, src->timestamp);
         break;
      default:
         unreachable("Unsupported query type");
      }

      if (!write_results)
         status = VK_NOT_READY;

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         cpu_write_query_result(dst, available_dst_idx, flags, available);
   }

   return status;
}

static VkResult
get_copy_queries_pipeline(struct hk_device *dev, VkPipelineLayout layout,
                          VkPipeline *pipeline_out)
{
   const char key[] = "hk-meta-copy-query-pool-results";
   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(key));
   if (cached != VK_NULL_HANDLE) {
      *pipeline_out = cached;
      return VK_SUCCESS;
   }

   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE,
      hk_get_nir_options(dev->vk.physical, MESA_SHADER_COMPUTE, NULL),
      "hk-meta-copy-queries");

   nir_builder *b = &build;

   nir_variable *push = nir_variable_create(b->shader, nir_var_mem_push_const,
                                            glsl_uint64_t_type(), "addr");
   libagx_copy_query(b, nir_load_var(b, push));
   agx_link_libagx(b->shader, dev->dev.libagx);

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = b->shader,
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &nir_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, key,
                                          sizeof(key), pipeline_out);
}

static void
hk_meta_copy_query_pool_results(struct hk_cmd_buffer *cmd,
                                struct hk_query_pool *pool,
                                uint32_t first_query, uint32_t query_count,
                                uint64_t dst_addr, uint64_t dst_stride,
                                VkQueryResultFlags flags)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   VkResult result;

   const struct libagx_copy_query_push info = {
      .availability = pool->bo->ptr.gpu,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->ptr.gpu
                                  : pool->bo->ptr.gpu + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->ptr.gpu + pool->query_start : 0,

      .first_query = first_query,
      .dst_addr = dst_addr,
      .dst_stride = dst_stride,

      .partial = flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
      ._64 = flags & VK_QUERY_RESULT_64_BIT,
      .with_availability = flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
   };

   uint64_t push = hk_pool_upload(cmd, &info, sizeof(info), 8);
   const char key[] = "hk-meta-copy-query-pool-results";
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(push),
   };
   VkPipelineLayout layout;
   result = vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL, &push_range,
                                        key, sizeof(key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline;
   result = get_copy_queries_pipeline(dev, layout, &pipeline);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   /* Save pipeline and push constants */
   struct hk_shader *shader_save = cmd->state.cs.shader;
   uint8_t push_save[HK_MAX_PUSH_SIZE];
   memcpy(push_save, desc->root.push, HK_MAX_PUSH_SIZE);

   dev->vk.dispatch_table.CmdBindPipeline(
      hk_cmd_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   vk_common_CmdPushConstants(hk_cmd_buffer_to_handle(cmd), layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                              &push);

   hk_CmdDispatchBase(hk_cmd_buffer_to_handle(cmd), 0, 0, 0, query_count, 1, 1);

   /* Restore pipeline and push constants */
   if (shader_save)
      hk_cmd_bind_compute_shader(cmd, shader_save);
   memcpy(desc->root.push, push_save, HK_MAX_PUSH_SIZE);
   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t firstQuery, uint32_t queryCount,
                           VkBuffer dstBuffer, VkDeviceSize dstOffset,
                           VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(hk_buffer, dst_buffer, dstBuffer);

   uint64_t dst_addr = hk_buffer_address(dst_buffer, dstOffset);
   hk_meta_copy_query_pool_results(cmd, pool, firstQuery, queryCount, dst_addr,
                                   stride, flags);
}
