/*
 * Copyright 2023 Valve Corporation
 * Copyright 2022 Collabora Ltd
 * SPDX-License-Identifier: MIT
 */

#include "compiler/glsl_types.h"
#include "compiler/shader_enums.h"
#include "mesa/program/prog_parameter.h"
#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "vulkan/vulkan_core.h"
#include "vk_buffer.h"
#include "vk_command_pool.h"
#include "vk_enum_to_str.h"
#include "vk_image.h"
#include "vk_meta.h"
#include "vk_meta_private.h"

#include "vk_command_buffer.h"
#include "vk_device.h"
#include "vk_format.h"
#include "vk_pipeline.h"

#include "nir.h"
#include "nir_builder.h"

#define BINDING_OUTPUT 0
#define BINDING_INPUT  1

static VkFormat
aspect_format(VkFormat fmt, VkImageAspectFlags aspect)
{
   bool depth = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT);
   bool stencil = (aspect & VK_IMAGE_ASPECT_STENCIL_BIT);

   enum pipe_format p_format = vk_format_to_pipe_format(fmt);

   if (util_format_is_depth_or_stencil(p_format)) {
      assert(depth ^ stencil);
      if (depth) {
         switch (fmt) {
         case VK_FORMAT_D32_SFLOAT:
         case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_FORMAT_D32_SFLOAT;
         case VK_FORMAT_D16_UNORM:
         case VK_FORMAT_D16_UNORM_S8_UINT:
            return VK_FORMAT_D16_UNORM;
         default:
            unreachable("invalid depth");
         }
      } else {
         switch (fmt) {
         case VK_FORMAT_S8_UINT:
         case VK_FORMAT_D32_SFLOAT_S8_UINT:
         case VK_FORMAT_D16_UNORM_S8_UINT:
            return VK_FORMAT_S8_UINT;
         default:
            unreachable("invalid stencil");
         }
      }
   }

   assert(!depth && !stencil);

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(fmt);

   if (ycbcr_info) {
      switch (aspect) {
      case VK_IMAGE_ASPECT_PLANE_0_BIT:
         return ycbcr_info->planes[0].format;
      case VK_IMAGE_ASPECT_PLANE_1_BIT:
         return ycbcr_info->planes[1].format;
      case VK_IMAGE_ASPECT_PLANE_2_BIT:
         return ycbcr_info->planes[2].format;
      default:
         unreachable("invalid ycbcr aspect");
      }
   }

   return fmt;
}

static VkFormat
canonical_format(VkFormat fmt)
{
   enum pipe_format p_format = vk_format_to_pipe_format(fmt);

   if (util_format_is_depth_or_stencil(p_format))
      return fmt;

   switch (util_format_get_blocksize(p_format)) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R16_UINT;
   case 4:
      return VK_FORMAT_R32_UINT;
   case 8:
      return VK_FORMAT_R32G32_UINT;
   case 16:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      unreachable("invalid bpp");
   }
}

enum copy_type {
   BUF2IMG,
   IMG2BUF,
   IMG2IMG,
};

struct vk_meta_push_data {
   uint32_t buffer_offset;
   uint32_t row_extent;
   uint32_t slice_or_layer_extent;

   int32_t src_offset_el[4];
   int32_t dst_offset_el[4];
} PACKED;

#define get_push(b, name)                                                      \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct vk_meta_push_data *)0)->name) * 8,                 \
      nir_imm_int(b, offsetof(struct vk_meta_push_data, name)))

struct vk_meta_buffer_copy_key {
   enum vk_meta_object_key_type key_type;
   enum copy_type type;
   unsigned block_size;
   unsigned nr_samples;
};

static nir_def *
linearize_coords(nir_builder *b, nir_def *coord,
                 const struct vk_meta_buffer_copy_key *key)
{
   assert(key->nr_samples == 1 && "buffer<-->image copies not multisampled");

   nir_def *row_extent = get_push(b, row_extent);
   nir_def *slice_or_layer_extent = get_push(b, slice_or_layer_extent);
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = nir_channel(b, coord, 1);
   nir_def *z_or_layer = nir_channel(b, coord, 2);

   nir_def *v = get_push(b, buffer_offset);

   v = nir_iadd(b, v, nir_imul_imm(b, x, key->block_size));
   v = nir_iadd(b, v, nir_imul(b, y, row_extent));
   v = nir_iadd(b, v, nir_imul(b, z_or_layer, slice_or_layer_extent));

   return nir_udiv_imm(b, v, key->block_size);
}

static nir_shader *
build_buffer_copy_shader(const struct vk_meta_buffer_copy_key *key)
{
   nir_builder build =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "vk-meta-copy");

   nir_builder *b = &build;

   bool src_is_buf = key->type == BUF2IMG;
   bool dst_is_buf = key->type == IMG2BUF;

   bool msaa = key->nr_samples > 1;
   enum glsl_sampler_dim dim_2d =
      msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
   enum glsl_sampler_dim dim_src = src_is_buf ? GLSL_SAMPLER_DIM_BUF : dim_2d;
   enum glsl_sampler_dim dim_dst = dst_is_buf ? GLSL_SAMPLER_DIM_BUF : dim_2d;

   /* Using uint types here is technically wrong but AGX ignores it. If other
    * backends care this needs to be added to the shader key :-(
    */
   const struct glsl_type *texture_type =
      glsl_sampler_type(dim_src, false, !src_is_buf, GLSL_TYPE_UINT);

   const struct glsl_type *image_type =
      glsl_image_type(dim_dst, !dst_is_buf, GLSL_TYPE_UINT);

   nir_variable *texture =
      nir_variable_create(b->shader, nir_var_uniform, texture_type, "source");
   nir_variable *image =
      nir_variable_create(b->shader, nir_var_image, image_type, "dest");

   image->data.descriptor_set = 0;
   image->data.binding = BINDING_OUTPUT;
   image->data.access = ACCESS_NON_READABLE;

   texture->data.descriptor_set = 0;
   texture->data.binding = BINDING_INPUT;

   /* Grab the offset vectors */
   nir_def *src_offset_el = nir_load_push_constant(
      b, 3, 32,
      nir_imm_int(b, offsetof(struct vk_meta_push_data, src_offset_el)));

   nir_def *dst_offset_el = nir_load_push_constant(
      b, 3, 32,
      nir_imm_int(b, offsetof(struct vk_meta_push_data, dst_offset_el)));

   /* We're done setting up variables, do the copy */
   nir_def *coord = nir_load_global_invocation_id(b, 32);

   nir_def *src_coord = nir_iadd(b, coord, src_offset_el);
   nir_def *dst_coord = nir_iadd(b, coord, dst_offset_el);

   /* Special case handle buffer indexing */
   if (dst_is_buf) {
      dst_coord = linearize_coords(b, coord, key);
   } else if (src_is_buf) {
      src_coord = linearize_coords(b, coord, key);
   }

   /* Copy formatted texel from texture to storage image */
   for (unsigned s = 0; s < key->nr_samples; ++s) {
      nir_deref_instr *deref = nir_build_deref_var(b, texture);
      nir_def *ms_index = nir_imm_int(b, s);

      nir_def *value = msaa ? nir_txf_ms_deref(b, deref, src_coord, ms_index)
                            : nir_txf_deref(b, deref, src_coord, NULL);

      nir_image_deref_store(b, &nir_build_deref_var(b, image)->def,
                            nir_pad_vec4(b, dst_coord), ms_index, value,
                            nir_imm_int(b, 0), .image_dim = dim_dst,
                            .image_array = !dst_is_buf);
   }

   return b->shader;
}

static VkResult
get_buffer_copy_descriptor_set_layout(struct vk_device *device,
                                      struct vk_meta_device *meta,
                                      VkDescriptorSetLayout *layout_out,
                                      enum copy_type type)
{
   const char *keys[] = {
      [IMG2BUF] = "vk-meta-copy-image-to-buffer-descriptor-set-layout",
      [BUF2IMG] = "vk-meta-copy-buffer-to-image-descriptor-set-layout",
      [IMG2IMG] = "vk-meta-copy-image-to-image-descriptor-set-layout",
   };

   VkDescriptorSetLayout from_cache = vk_meta_lookup_descriptor_set_layout(
      meta, keys[type], strlen(keys[type]));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = BINDING_OUTPUT,
         .descriptorType = type != IMG2BUF
                              ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                              : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = BINDING_INPUT,
         .descriptorType = type == BUF2IMG
                              ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                              : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };

   return vk_meta_create_descriptor_set_layout(device, meta, &info, keys[type],
                                               strlen(keys[type]), layout_out);
}

static VkResult
get_buffer_copy_pipeline_layout(struct vk_device *device,
                                struct vk_meta_device *meta,
                                struct vk_meta_buffer_copy_key *key,
                                VkDescriptorSetLayout set_layout,
                                VkPipelineLayout *layout_out,
                                enum copy_type type)
{
   const char *keys[] = {
      [IMG2BUF] = "vk-meta-copy-image-to-buffer-pipeline-layout",
      [BUF2IMG] = "vk-meta-copy-buffer-to-image-pipeline-layout",
      [IMG2IMG] = "vk-meta-copy-image-to-image-pipeline-layout",
   };

   VkPipelineLayout from_cache =
      vk_meta_lookup_pipeline_layout(meta, keys[type], strlen(keys[type]));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   VkPipelineLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };

   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(struct vk_meta_push_data),
   };

   info.pushConstantRangeCount = 1;
   info.pPushConstantRanges = &push_range;

   return vk_meta_create_pipeline_layout(device, meta, &info, keys[type],
                                         strlen(keys[type]), layout_out);
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

void
vk_meta_copy_image_to_buffer2(struct vk_command_buffer *cmd,
                              struct vk_meta_device *meta,
                              const VkCopyImageToBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(vk_image, image, pCopyBufferInfo->srcImage);
   VK_FROM_HANDLE(vk_image, src_image, pCopyBufferInfo->srcImage);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkResult result;

   VkDescriptorSetLayout set_layout;
   result =
      get_buffer_copy_descriptor_set_layout(device, meta, &set_layout, IMG2BUF);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer =
      util_format_is_compressed(vk_format_to_pipe_format(image->format));

   for (unsigned i = 0; i < pCopyBufferInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *region = &pCopyBufferInfo->pRegions[i];

      unsigned layers = MAX2(region->imageExtent.depth,
                             vk_image_subresource_layer_count(
                                src_image, &region->imageSubresource));
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {

         VkImageAspectFlags aspect = region->imageSubresource.aspectMask;
         VkFormat aspect_fmt = aspect_format(image->format, aspect);
         VkFormat canonical = canonical_format(aspect_fmt);

         uint32_t blocksize_B =
            util_format_get_blocksize(vk_format_to_pipe_format(canonical));

#if 0
      printf("img2buf %ux%ux%u\n", region->imageExtent.width,
             region->imageExtent.height, region->imageExtent.depth);
#endif

      enum pipe_format p_format = vk_format_to_pipe_format(image->format);

      unsigned row_extent =
         util_format_get_nblocksx(p_format, MAX2(region->bufferRowLength,
                                                 region->imageExtent.width)) *
         blocksize_B;
      unsigned slice_extent =
         util_format_get_nblocksy(p_format, MAX2(region->bufferImageHeight,
                                                 region->imageExtent.height)) *
         row_extent;
      unsigned layer_extent =
         util_format_get_nblocksz(p_format, region->imageExtent.depth) *
         slice_extent;

      bool is_3d = region->imageExtent.depth > 1;

      struct vk_meta_buffer_copy_key key = {
         .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER_PIPELINE,
         .type = IMG2BUF,
         .block_size = blocksize_B,
         .nr_samples = image->samples,
      };

      VkPipelineLayout pipeline_layout;
      result = get_buffer_copy_pipeline_layout(device, meta, &key, set_layout,
                                               &pipeline_layout, false);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      VkImageView src_view;
      const VkImageViewUsageCreateInfo src_view_usage = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      const VkImageViewCreateInfo src_view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .flags = VK_IMAGE_VIEW_CREATE_INTERNAL_MESA,
         .pNext = &src_view_usage,
         .image = pCopyBufferInfo->srcImage,
         .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
         .format = canonical,
         .subresourceRange =
            {
               .aspectMask = region->imageSubresource.aspectMask,
               .baseMipLevel = region->imageSubresource.mipLevel,
               .baseArrayLayer = MAX2(region->imageOffset.z,
                                      region->imageSubresource.baseArrayLayer) +
                                 layer_offs,
               .layerCount = per_layer ? 1 : layers,
               .levelCount = 1,
            },
      };

      result = vk_meta_create_image_view(cmd, meta, &src_view_info, &src_view);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      VkDescriptorImageInfo src_info = {
         .imageLayout = pCopyBufferInfo->srcImageLayout,
         .imageView = src_view,
      };

      VkWriteDescriptorSet desc_writes[2];

      const VkBufferViewCreateInfo dst_view_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = pCopyBufferInfo->dstBuffer,
         .format = canonical,

         /* Ideally, this would be region->bufferOffset, but that might not
          * be aligned to minTexelBufferOffsetAlignment. Instead, we use a 0
          * offset (which is definitely aligned) and add the offset ourselves
          * in the shader.
          */
         .offset = 0,
         .range = VK_WHOLE_SIZE,
      };

      VkBufferView dst_view;
      VkResult result =
         vk_meta_create_buffer_view(cmd, meta, &dst_view_info, &dst_view);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      desc_writes[0] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = 0,
         .dstBinding = BINDING_OUTPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
         .descriptorCount = 1,
         .pTexelBufferView = &dst_view,
      };

      desc_writes[1] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = 0,
         .dstBinding = BINDING_INPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .pImageInfo = &src_info,
      };

      disp->CmdPushDescriptorSetKHR(
         vk_command_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_COMPUTE,
         pipeline_layout, 0, ARRAY_SIZE(desc_writes), desc_writes);

      VkPipeline pipeline;
      result = get_buffer_copy_pipeline(device, meta, &key, pipeline_layout,
                                        &pipeline);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                            VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

      enum pipe_format p_src_fmt = vk_format_to_pipe_format(src_image->format);

      struct vk_meta_push_data push = {
         .buffer_offset = region->bufferOffset,
         .row_extent = row_extent,
         .slice_or_layer_extent = is_3d ? slice_extent : layer_extent,

         .src_offset_el[0] =
            util_format_get_nblocksx(p_src_fmt, region->imageOffset.x),
         .src_offset_el[1] =
            util_format_get_nblocksy(p_src_fmt, region->imageOffset.y),

      };

      push.buffer_offset += push.slice_or_layer_extent * layer_offs;

      disp->CmdPushConstants(vk_command_buffer_to_handle(cmd), pipeline_layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                             &push);

      /* Dispatch 1 thread per texel */
      disp->CmdDispatch(
         vk_command_buffer_to_handle(cmd),
         util_format_get_nblocksx(p_format, region->imageExtent.width),
         util_format_get_nblocksy(p_format, region->imageExtent.height),
         per_layer ? 1 : layers);
      }
   }
}

void
vk_meta_copy_buffer_to_image2(struct vk_command_buffer *cmd,
                              struct vk_meta_device *meta,
                              const struct VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(vk_image, image, info->dstImage);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkDescriptorSetLayout set_layout;
   VkResult result =
      get_buffer_copy_descriptor_set_layout(device, meta, &set_layout, BUF2IMG);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer =
      util_format_is_compressed(vk_format_to_pipe_format(image->format));

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];

      unsigned layers = MAX2(
         region->imageExtent.depth,
         vk_image_subresource_layer_count(image, &region->imageSubresource));
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {
         VkImageAspectFlags aspect = region->imageSubresource.aspectMask;
         VkFormat aspect_fmt = aspect_format(image->format, aspect);
         VkFormat canonical = canonical_format(aspect_fmt);
         enum pipe_format p_format = vk_format_to_pipe_format(aspect_fmt);
         uint32_t blocksize_B = util_format_get_blocksize(p_format);

#if 0
      printf("buf2img %ux%ux%u as %s\n", region->imageExtent.width,
             region->imageExtent.height, region->imageExtent.depth,
             util_format_short_name(vk_format_to_pipe_format(canonical)));
#endif

      bool is_3d = region->imageExtent.depth > 1;

      struct vk_meta_buffer_copy_key key = {
         .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER_PIPELINE,
         .type = BUF2IMG,
         .block_size = blocksize_B,
         .nr_samples = image->samples,
      };

      VkPipelineLayout pipeline_layout;
      result = get_buffer_copy_pipeline_layout(device, meta, &key, set_layout,
                                               &pipeline_layout, true);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      VkWriteDescriptorSet desc_writes[2];

      unsigned row_extent =
         util_format_get_nblocksx(p_format, MAX2(region->bufferRowLength,
                                                 region->imageExtent.width)) *
         blocksize_B;
      unsigned slice_extent =
         util_format_get_nblocksy(p_format, MAX2(region->bufferImageHeight,
                                                 region->imageExtent.height)) *
         row_extent;
      unsigned layer_extent =
         util_format_get_nblocksz(p_format, region->imageExtent.depth) *
         slice_extent;

      /* Create a view into the source buffer as a texel buffer */
      const VkBufferViewCreateInfo src_view_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = info->srcBuffer,
         .format = canonical,

         /* Ideally, this would be region->bufferOffset, but that might not be
          * aligned to minTexelBufferOffsetAlignment. Instead, we use a 0 offset
          * (which is definitely aligned) and add the offset ourselves in the
          * shader.
          */
         .offset = 0,
         .range = VK_WHOLE_SIZE,
      };

      assert((region->bufferOffset % blocksize_B) == 0 && "must be aligned");

      VkBufferView src_view;
      result = vk_meta_create_buffer_view(cmd, meta, &src_view_info, &src_view);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      VkImageView dst_view;
      const VkImageViewUsageCreateInfo dst_view_usage = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      };
      const VkImageViewCreateInfo dst_view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .flags = VK_IMAGE_VIEW_CREATE_INTERNAL_MESA,
         .pNext = &dst_view_usage,
         .image = info->dstImage,
         .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
         .format = canonical,
         .subresourceRange =
            {
               .aspectMask = region->imageSubresource.aspectMask,
               .baseMipLevel = region->imageSubresource.mipLevel,
               .baseArrayLayer = MAX2(region->imageOffset.z,
                                      region->imageSubresource.baseArrayLayer) +
                                 layer_offs,
               .layerCount = per_layer ? 1 : layers,
               .levelCount = 1,
            },
      };

      result = vk_meta_create_image_view(cmd, meta, &dst_view_info, &dst_view);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      const VkDescriptorImageInfo dst_info = {
         .imageView = dst_view,
         .imageLayout = info->dstImageLayout,
      };

      desc_writes[0] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = 0,
         .dstBinding = BINDING_OUTPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .pImageInfo = &dst_info,
      };

      desc_writes[1] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = 0,
         .dstBinding = BINDING_INPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .descriptorCount = 1,
         .pTexelBufferView = &src_view,
      };

      disp->CmdPushDescriptorSetKHR(
         vk_command_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_COMPUTE,
         pipeline_layout, 0, ARRAY_SIZE(desc_writes), desc_writes);

      VkPipeline pipeline;
      result = get_buffer_copy_pipeline(device, meta, &key, pipeline_layout,
                                        &pipeline);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(cmd, result);
         return;
      }

      disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                            VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

      struct vk_meta_push_data push = {
         .buffer_offset = region->bufferOffset,
         .row_extent = row_extent,
         .slice_or_layer_extent = is_3d ? slice_extent : layer_extent,

         .dst_offset_el[0] =
            util_format_get_nblocksx(p_format, region->imageOffset.x),
         .dst_offset_el[1] =
            util_format_get_nblocksy(p_format, region->imageOffset.y),
      };

      push.buffer_offset += push.slice_or_layer_extent * layer_offs;

      disp->CmdPushConstants(vk_command_buffer_to_handle(cmd), pipeline_layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                             &push);

      /* Dispatch 1 thread per texel */
      disp->CmdDispatch(
         vk_command_buffer_to_handle(cmd),
         util_format_get_nblocksx(p_format, region->imageExtent.width),
         util_format_get_nblocksy(p_format, region->imageExtent.height),
         per_layer ? 1 : layers);
      }
   }
}

void
vk_meta_copy_image2(struct vk_command_buffer *cmd, struct vk_meta_device *meta,
                    const struct VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(vk_image, src_image, info->srcImage);
   VK_FROM_HANDLE(vk_image, dst_image, info->dstImage);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkDescriptorSetLayout set_layout;
   VkResult result =
      get_buffer_copy_descriptor_set_layout(device, meta, &set_layout, BUF2IMG);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer =
      util_format_is_compressed(vk_format_to_pipe_format(src_image->format)) ||
      util_format_is_compressed(vk_format_to_pipe_format(dst_image->format));

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkImageCopy2 *region = &info->pRegions[r];

      unsigned layers = MAX2(
         vk_image_subresource_layer_count(src_image, &region->srcSubresource),
         region->extent.depth);
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {
         u_foreach_bit(aspect, region->srcSubresource.aspectMask) {
            /* We use the source format throughout for consistent scaling with
             * compressed<-->uncompressed copies, where the extents are defined
             * to follow the source.
             */
            VkFormat aspect_fmt = aspect_format(src_image->format, 1 << aspect);
            VkFormat canonical = canonical_format(aspect_fmt);
            uint32_t blocksize_B =
               util_format_get_blocksize(vk_format_to_pipe_format(canonical));

#if 0
         printf(
            "img2img %ux%ux%u, %s, %s as %s\n", region->extent.width,
            region->extent.height, region->extent.depth,
            util_format_short_name(vk_format_to_pipe_format(src_image->format)),
            util_format_short_name(vk_format_to_pipe_format(dst_image->format)),
            util_format_short_name(vk_format_to_pipe_format(canonical)));
#endif

         struct vk_meta_buffer_copy_key key = {
            .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER_PIPELINE,
            .type = IMG2IMG,
            .block_size = blocksize_B,
            .nr_samples = dst_image->samples,
         };

         assert(key.nr_samples == src_image->samples);

         VkPipelineLayout pipeline_layout;
         result = get_buffer_copy_pipeline_layout(
            device, meta, &key, set_layout, &pipeline_layout, true);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         VkWriteDescriptorSet desc_writes[2];

         VkImageView src_view;
         const VkImageViewUsageCreateInfo src_view_usage = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
         };
         const VkImageViewCreateInfo src_view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = VK_IMAGE_VIEW_CREATE_INTERNAL_MESA,
            .pNext = &src_view_usage,
            .image = info->srcImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = canonical,
            .subresourceRange =
               {
                  .aspectMask =
                     region->srcSubresource.aspectMask & (1 << aspect),
                  .baseMipLevel = region->srcSubresource.mipLevel,
                  .baseArrayLayer =
                     MAX2(region->srcOffset.z,
                          region->srcSubresource.baseArrayLayer) +
                     layer_offs,
                  .layerCount = per_layer ? 1 : layers,
                  .levelCount = 1,
               },
         };

         result =
            vk_meta_create_image_view(cmd, meta, &src_view_info, &src_view);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         VkDescriptorImageInfo src_info = {
            .imageLayout = info->srcImageLayout,
            .imageView = src_view,
         };

         VkImageView dst_view;
         const VkImageViewUsageCreateInfo dst_view_usage = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
         };
         const VkImageViewCreateInfo dst_view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = VK_IMAGE_VIEW_CREATE_INTERNAL_MESA,
            .pNext = &dst_view_usage,
            .image = info->dstImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = canonical,
            .subresourceRange =
               {
                  .aspectMask =
                     vk_format_get_ycbcr_info(dst_image->format) ||
                           vk_format_get_ycbcr_info(src_image->format)
                        ? region->dstSubresource.aspectMask
                        : (1 << aspect),
                  .baseMipLevel = region->dstSubresource.mipLevel,
                  .baseArrayLayer =
                     MAX2(region->dstOffset.z,
                          region->dstSubresource.baseArrayLayer) +
                     layer_offs,
                  .layerCount = per_layer ? 1 : layers,
                  .levelCount = 1,
               },
         };

         result =
            vk_meta_create_image_view(cmd, meta, &dst_view_info, &dst_view);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         const VkDescriptorImageInfo dst_info = {
            .imageView = dst_view,
            .imageLayout = info->dstImageLayout,
         };

         desc_writes[0] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = 0,
            .dstBinding = BINDING_OUTPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .pImageInfo = &dst_info,
         };

         desc_writes[1] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = 0,
            .dstBinding = BINDING_INPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = 1,
            .pImageInfo = &src_info,
         };

         disp->CmdPushDescriptorSetKHR(
            vk_command_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, ARRAY_SIZE(desc_writes), desc_writes);

         VkPipeline pipeline;
         result = get_buffer_copy_pipeline(device, meta, &key, pipeline_layout,
                                           &pipeline);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                               VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

         enum pipe_format p_src_fmt =
            vk_format_to_pipe_format(src_image->format);
         enum pipe_format p_dst_fmt =
            vk_format_to_pipe_format(dst_image->format);

         struct vk_meta_push_data push = {
            .src_offset_el[0] =
               util_format_get_nblocksx(p_src_fmt, region->srcOffset.x),
            .src_offset_el[1] =
               util_format_get_nblocksy(p_src_fmt, region->srcOffset.y),

            .dst_offset_el[0] =
               util_format_get_nblocksx(p_dst_fmt, region->dstOffset.x),
            .dst_offset_el[1] =
               util_format_get_nblocksy(p_dst_fmt, region->dstOffset.y),
         };

         disp->CmdPushConstants(vk_command_buffer_to_handle(cmd),
                                pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(push), &push);

         /* Dispatch 1 thread per texel */
         enum pipe_format p_format = vk_format_to_pipe_format(aspect_fmt);
         disp->CmdDispatch(
            vk_command_buffer_to_handle(cmd),
            util_format_get_nblocksx(p_format, region->extent.width),
            util_format_get_nblocksy(p_format, region->extent.height),
            per_layer ? 1 : layers);
         }
      }
   }
}
