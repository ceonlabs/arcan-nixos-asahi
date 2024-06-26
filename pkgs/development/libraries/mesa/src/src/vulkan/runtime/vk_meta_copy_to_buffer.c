/*
 * Copyright 2023 Valve Corporation
 * Copyright 2022 Collabora Ltd
 * SPDX-License-Identifier: MIT
 */

#include "compiler/glsl_types.h"
#include "compiler/shader_enums.h"
#include "mesa/program/prog_parameter.h"
#include "util/bitscan.h"
#include "util/u_math.h"
#include "vulkan/vulkan_core.h"
#include "vk_buffer.h"
#include "vk_command_pool.h"
#include "vk_meta_private.h"

#include "vk_command_buffer.h"
#include "vk_device.h"
#include "vk_format.h"
#include "vk_pipeline.h"

#include "nir.h"
#include "nir_builder.h"

#define BINDING_OUTPUT 0
#define BINDING_INPUT  1

struct vk_meta_image_to_buffer_push_data {
   uint32_t dest_offset_el;
};

#define get_image_push(b, name)                                                \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct vk_meta_image_to_buffer_push_data *)0)->name) * 8, \
      nir_imm_int(b,                                                           \
                  offsetof(struct vk_meta_image_to_buffer_push_data, name)))

enum copy_source {
   COPY_SOURCE_PATTERN,
   COPY_SOURCE_BUFFER,
};

struct vk_meta_buffer_copy_key {
   enum vk_meta_object_key_type key_type;
   enum copy_source source;

   /* Power-of-two block size for the transfer, range [1, 16] */
   uint8_t blocksize;
   uint8_t pad[3];
};
static_assert(sizeof(struct vk_meta_buffer_copy_key) == 12, "packed");

/* XXX: TODO: move to common */
/* Copyright © Microsoft Corporation */
static nir_def *
dzn_nir_create_bo_desc(nir_builder *b, nir_variable_mode mode,
                       uint32_t desc_set, uint32_t binding, const char *name,
                       unsigned access, const struct glsl_type *dummy_type)
{
   nir_variable *var = nir_variable_create(b->shader, mode, dummy_type, name);
   var->data.descriptor_set = desc_set;
   var->data.binding = binding;
   var->data.access = access;

   assert(mode == nir_var_mem_ubo || mode == nir_var_mem_ssbo);
   if (mode == nir_var_mem_ubo)
      b->shader->info.num_ubos++;
   else
      b->shader->info.num_ssbos++;

   VkDescriptorType desc_type = var->data.mode == nir_var_mem_ubo
                                   ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                   : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   nir_address_format addr_format = nir_address_format_64bit_global_32bit_offset; /* XXX */
   nir_def *index = nir_vulkan_resource_index(
      b, nir_address_format_num_components(addr_format),
      nir_address_format_bit_size(addr_format), nir_imm_int(b, 0),
      .desc_set = desc_set, .binding = binding, .desc_type = desc_type);

   nir_def *desc = nir_load_vulkan_descriptor(
      b, nir_address_format_num_components(addr_format),
      nir_address_format_bit_size(addr_format), index, .desc_type = desc_type);

   return desc;
}

static const struct glsl_type *
type_for_blocksize(uint8_t blocksize)
{
   assert(util_is_power_of_two_nonzero(blocksize) && blocksize <= 16);

   if (blocksize > 4)
      return glsl_vector_type(GLSL_TYPE_UINT, blocksize / 4);
   else
      return glsl_uintN_t_type(8 * blocksize);
}

static nir_shader *
build_buffer_copy_shader(const struct vk_meta_buffer_copy_key *key)
{
   nir_builder build = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL,
                                                      "vk-meta-copy-to-buffer");
   nir_builder *b = &build;

   const struct glsl_type *type =
      glsl_array_type(type_for_blocksize(key->blocksize), 0, key->blocksize);

   nir_def *index = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   nir_def *value;

   if (key->source == COPY_SOURCE_BUFFER) {
      nir_def *ubo =
         dzn_nir_create_bo_desc(b, nir_var_mem_ubo, 0, BINDING_INPUT, "source",
                                ACCESS_NON_WRITEABLE, type);
      nir_deref_instr *ubo_deref =
         nir_build_deref_cast(b, ubo, nir_var_mem_ubo, type, key->blocksize);

      nir_deref_instr *element_deref = nir_build_deref_array(
         b, ubo_deref, nir_u2uN(b, index, ubo_deref->def.bit_size));

      value = nir_load_deref(b, element_deref);
   } else {
      nir_def *pattern =
         nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0));

      assert(key->blocksize >= 4 && "fills at least 32-bit");
      value = nir_replicate(b, pattern, key->blocksize / 4);
   }

   /* Write out raw bytes to SSBO */
   nir_def *ssbo =
      dzn_nir_create_bo_desc(b, nir_var_mem_ssbo, 0, BINDING_OUTPUT,
                             "destination", ACCESS_NON_READABLE, type);

   nir_deref_instr *ssbo_deref =
      nir_build_deref_cast(b, ssbo, nir_var_mem_ssbo, type, key->blocksize);

   nir_deref_instr *element_deref = nir_build_deref_array(
      b, ssbo_deref, nir_u2uN(b, index, ssbo_deref->def.bit_size));

   nir_store_deref(b, element_deref, value,
                   nir_component_mask(value->num_components));

   return b->shader;
}

static VkResult
get_buffer_copy_descriptor_set_layout(struct vk_device *device,
                                      struct vk_meta_device *meta,
                                      VkDescriptorSetLayout *layout_out,
                                      enum copy_source source)
{
   const char buffer_key[] = "vk-meta-buffer-copy-descriptor-set-layout";
   const char fill_key[] = "vk-meta-fill__-copy-descriptor-set-layout";

   static_assert(sizeof(buffer_key) == sizeof(fill_key));
   const char *key = source == COPY_SOURCE_BUFFER ? buffer_key : fill_key;

   VkDescriptorSetLayout from_cache =
      vk_meta_lookup_descriptor_set_layout(meta, key, sizeof(buffer_key));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = BINDING_OUTPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = BINDING_INPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };

   return vk_meta_create_descriptor_set_layout(device, meta, &info, key,
                                               sizeof(key), layout_out);
}

static VkResult
get_buffer_copy_pipeline_layout(struct vk_device *device,
                                struct vk_meta_device *meta,
                                struct vk_meta_buffer_copy_key *key,
                                VkDescriptorSetLayout set_layout,
                                VkPipelineLayout *layout_out)
{
   const char copy_key[] = "vk-meta-buffer-copy-pipeline-layout";
   const char fill_key[] = "vk-meta-buffer-fill-pipeline-layout";
   const char cimg_key[] = "vk-meta-buffer-cimg-pipeline-layout";

   STATIC_ASSERT(sizeof(copy_key) == sizeof(fill_key));
   STATIC_ASSERT(sizeof(copy_key) == sizeof(cimg_key));
   const char *pipeline_key =
      key->source == COPY_SOURCE_BUFFER ? copy_key : fill_key;

   VkPipelineLayout from_cache =
      vk_meta_lookup_pipeline_layout(meta, pipeline_key, sizeof(copy_key));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   VkPipelineLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };

   size_t push_size = 0;
   if (key->source == COPY_SOURCE_PATTERN)
      push_size = sizeof(uint32_t);

   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_size,
   };

   if (push_size) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &push_range;
   }

   return vk_meta_create_pipeline_layout(device, meta, &info, pipeline_key,
                                         sizeof(copy_key), layout_out);
}

static VkResult
get_buffer_copy_pipeline(struct vk_device *device, struct vk_meta_device *meta,
                         const struct vk_meta_buffer_copy_key *key,
                         VkPipelineLayout layout, VkPipeline *pipeline_out)
{
   VkPipeline from_cache = vk_meta_lookup_pipeline(meta, key, sizeof(*key));
   if (from_cache != VK_NULL_HANDLE) {
      *pipeline_out = from_cache;
      return VK_SUCCESS;
   }

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_buffer_copy_shader(key),
   };
   const VkPipelineShaderStageCreateInfo cs_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &nir_info,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .pName = "main",
   };

   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = cs_info,
      .layout = layout,
   };

   VkResult result = vk_meta_create_compute_pipeline(
      device, meta, &info, key, sizeof(*key), pipeline_out);
   ralloc_free(nir_info.nir);

   return result;
}

static unsigned
alignment_of(unsigned x)
{
   return 1 << MIN2(__builtin_ctz(x), 31);
}

struct copy_desc {
   enum copy_source source;

   union {
      uint32_t pattern;

      struct {
         struct vk_buffer *source;
         VkDeviceSize srcOffset;
      } buffer;

      struct {
         struct vk_image *image;
         VkDescriptorImageInfo *info;
         VkFormat format;
         struct vk_meta_image_to_buffer_push_data push;
      } image;
   };
};

static void
do_copy(struct vk_command_buffer *cmd, struct vk_meta_device *meta, size_t size,
        struct vk_buffer *dest, VkDeviceSize dstOffset, struct copy_desc *desc)
{
   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;
   VkResult result;

   /* The "alignment" of the copy is the maximum alignment that all accesses
    * within the copy will satsify.
    */
   unsigned alignment = MIN2(alignment_of(dstOffset), alignment_of(size));

   if (desc->source == COPY_SOURCE_BUFFER)
      alignment = MIN2(alignment, alignment_of(desc->buffer.srcOffset));

   struct vk_meta_buffer_copy_key key = {
      .key_type = VK_META_OBJECT_KEY_FILL_PIPELINE,
      .source = desc->source,
      .blocksize = MIN2(alignment, 16),
   };

   VkDescriptorSetLayout set_layout;
   result = get_buffer_copy_descriptor_set_layout(device, meta, &set_layout,
                                                  desc->source);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   VkPipelineLayout pipeline_layout;
   result = get_buffer_copy_pipeline_layout(device, meta, &key, set_layout,
                                            &pipeline_layout);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   VkDescriptorBufferInfo buffer_infos[2];
   VkWriteDescriptorSet desc_writes[2];

   for (unsigned i = 0; i < 2; ++i) {
      bool is_dest = (i == BINDING_OUTPUT);

      if (!is_dest && desc->source != COPY_SOURCE_BUFFER)
         continue;

      buffer_infos[i] = (VkDescriptorBufferInfo){
         .buffer = vk_buffer_to_handle(is_dest ? dest : desc->buffer.source),
         .offset = is_dest ? dstOffset : desc->buffer.srcOffset,
         .range = size,
      };

      desc_writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = 0,
         .dstBinding = i,
         .descriptorType = is_dest ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                   : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .pBufferInfo = &buffer_infos[i],
      };
   }

   unsigned desc_count = desc->source == COPY_SOURCE_PATTERN ? 1 : 2;
   disp->CmdPushDescriptorSetKHR(vk_command_buffer_to_handle(cmd),
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline_layout, 0, desc_count, desc_writes);

   VkPipeline pipeline;
   result =
      get_buffer_copy_pipeline(device, meta, &key, pipeline_layout, &pipeline);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                         VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   if (desc->source == COPY_SOURCE_PATTERN) {
      disp->CmdPushConstants(vk_command_buffer_to_handle(cmd), pipeline_layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
                             &desc->pattern);
   }

   disp->CmdDispatch(vk_command_buffer_to_handle(cmd), size / key.blocksize, 1,
                     1);
}

void
vk_meta_fill_buffer(struct vk_command_buffer *cmd, struct vk_meta_device *meta,
                    struct vk_buffer *dest, VkDeviceSize dstOffset,
                    VkDeviceSize dstRange, uint32_t data)
{
   size_t size = ROUND_DOWN_TO(vk_buffer_range(dest, dstOffset, dstRange), 4);
   dstOffset = ROUND_DOWN_TO(dstOffset, 4);

   do_copy(cmd, meta, size, dest, dstOffset,
           &(struct copy_desc){
              .source = COPY_SOURCE_PATTERN,
              .pattern = data,
           });
}

void
vk_meta_update_buffer(struct vk_command_buffer *cmd,
                      struct vk_meta_device *meta, struct vk_buffer *dest,
                      VkDeviceSize dstOffset, VkDeviceSize dstRange,
                      const void *data)
{
   /* Create a buffer to hold the data */
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = vk_buffer_range(dest, dstOffset, dstRange),
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &cmd->pool->queue_family_index,
   };

   VkBuffer buffer;
   VkResult result = vk_meta_create_buffer(cmd, meta, &info, &buffer);
   if (unlikely(result != VK_SUCCESS))
      return;

   /* Map the buffer for CPU access */
   void *map;
   result = meta->cmd_bind_map_buffer(cmd, meta, buffer, &map);
   if (unlikely(result != VK_SUCCESS))
      return;

   /* Copy from the CPU input to the staging buffer */
   memcpy(map, data, info.size);

   /* Copy between the buffers on the GPU */
   VK_FROM_HANDLE(vk_buffer, buffer_, buffer);
   size_t size = ROUND_DOWN_TO(vk_buffer_range(dest, dstOffset, dstRange), 4);
   dstOffset = ROUND_DOWN_TO(dstOffset, 4);

   do_copy(cmd, meta, size, dest, dstOffset,
           &(struct copy_desc){
              .source = COPY_SOURCE_BUFFER,
              .buffer.source = buffer_,
           });
}

void
vk_meta_copy_buffer2(struct vk_command_buffer *cmd, struct vk_meta_device *meta,
                     const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(vk_buffer, dst, pCopyBufferInfo->dstBuffer);
   VK_FROM_HANDLE(vk_buffer, src, pCopyBufferInfo->srcBuffer);

   for (unsigned i = 0; i < pCopyBufferInfo->regionCount; ++i) {
      const VkBufferCopy2 *copy = &pCopyBufferInfo->pRegions[i];

      do_copy(cmd, meta, copy->size, dst, copy->dstOffset,
              &(struct copy_desc){
                 .source = COPY_SOURCE_BUFFER,
                 .buffer.source = src,
                 .buffer.srcOffset = copy->srcOffset,
              });
   }
}
