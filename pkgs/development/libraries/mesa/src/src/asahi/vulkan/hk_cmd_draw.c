/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include "agx_bg_eot.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_ppp.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "util/bitpack_helpers.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "nir_builder.h"
#include "nir_lower_blend.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_render_pass.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Construct the key */
   struct agx_bg_eot_key key = {.tib = render->tilebuffer};
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] && (partial_render || store))
         continue;

      if (store) {
         bool store = att_info->storeOp == VK_ATTACHMENT_STORE_OP_STORE;

         /* When resolving, we store the intermediate multisampled image as the
          * resolve is a separate control stream. This could be optimized.
          */
         store |= att_info->resolveMode != VK_RESOLVE_MODE_NONE;

         /* Partial renders always need to flush to memory. */
         store |= partial_render;

         key.op[i] = store ? AGX_EOT_STORE : AGX_BG_EOT_NONE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         key.op[i] = load    ? AGX_BG_LOAD
                     : clear ? AGX_BG_CLEAR
                             : AGX_BG_EOT_NONE;
      }
   }

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         uint32_t index = key.tib.layered
                             ? iview->planes[0].layered_background_desc_index
                             : iview->planes[0].background_desc_index;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = dev->images.bo->ptr.gpu + index * AGX_TEXTURE_LENGTH;
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         uint32_t index = key.tib.layered
                             ? iview->planes[0].layered_eot_pbe_desc_index
                             : iview->planes[0].eot_pbe_desc_index;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = dev->images.bo->ptr.gpu + index * AGX_TEXTURE_LENGTH;
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->rodata.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = shader->ptr;
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = shader->ptr + shader->info.preamble_offset;
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static bool
is_aligned(unsigned x, unsigned pot_alignment)
{
   assert(util_is_power_of_two_nonzero(pot_alignment));
   return (x & (pot_alignment - 1)) == 0;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);
   }
}

static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *attach_z,
                    const VkRenderingAttachmentInfo *attach_s,
                    bool incomplete_render_area, bool partial_render)
{
   agx_pack(packed, ZLS_CONTROL, zls_control) {
      if (z_layout) {
         zls_control.z_store_enable =
            attach_z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
            attach_z->resolveMode != VK_RESOLVE_MODE_NONE || partial_render;

         zls_control.z_load_enable =
            attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
            incomplete_render_area;

         if (ail_is_compressed(z_layout)) {
            zls_control.z_compress_1 = true;
            zls_control.z_compress_2 = true;
         }

         if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
            zls_control.z_format = AGX_ZLS_FORMAT_16;
         } else {
            zls_control.z_format = AGX_ZLS_FORMAT_32F;
         }
      }

      if (s_layout) {
         /* TODO:
          * Fail
          * dEQP-VK.renderpass.dedicated_allocation.formats.d32_sfloat_s8_uint.input.dont_care.store.self_dep_clear_draw_use_input_aspect
          * without the force
          * .. maybe a VkRenderPass emulation bug.
          */
         zls_control.s_store_enable =
            attach_s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
            attach_s->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
            true;

         zls_control.s_load_enable =
            attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
            incomplete_render_area;

         if (ail_is_compressed(s_layout)) {
            zls_control.s_compress_1 = true;
            zls_control.s_compress_2 = true;
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview);
   }

   hk_merge_render_iview(render,
                         render->depth_att.iview ?: render->stencil_att.iview);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = vk_format_to_pipe_format(render->color_att[i].vk_format);
   }

   /* For now, we force layered=true since it makes compatibility problems way
    * easier.
    */
   render->tilebuffer = agx_build_tilebuffer_layout(
      formats, render->color_att_count, render->tilebuffer.nr_samples, true);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area);
   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true, incomplete_render_area);

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false, incomplete_render_area);
   render->cr.eot.partial = render->cr.eot.main;

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   render->cr.iogpu_unk_214 = 0xc000;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(vk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (ail_is_compressed(z_layout)) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(
            is_aligned(z_layout->compression_layer_stride_B, AIL_CACHELINE) &&
            "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
         render->cr.iogpu_unk_214 |= 0x40000;
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;

      if (ail_is_compressed(s_layout)) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(
            is_aligned(s_layout->compression_layer_stride_B, AIL_CACHELINE) &&
            "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(&cmd->current_cs.post_gfx);

   /* Don't reorder compute across render passes.
    *
    * TODO: Check if this is necessary if the proper PipelineBarriers are
    * handled... there may be CTS bugs...
    */
   hk_cmd_buffer_end_compute(cmd);

   if (incomplete_render_area) {
      uint32_t clear_count = 0;
      VkClearAttachment clear_att[HK_MAX_RTS + 1];
      for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
         const VkRenderingAttachmentInfo *att_info =
            &pRenderingInfo->pColorAttachments[i];
         if (att_info->imageView == VK_NULL_HANDLE ||
             att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
            continue;

         clear_att[clear_count++] = (VkClearAttachment){
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = i,
            .clearValue = att_info->clearValue,
         };
      }

      clear_att[clear_count] = (VkClearAttachment){
         .aspectMask = 0,
      };

      if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
          attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
         clear_att[clear_count].clearValue.depthStencil.depth =
            attach_z->clearValue.depthStencil.depth;
      }

      if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
          attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
         clear_att[clear_count].clearValue.depthStencil.stencil =
            attach_s->clearValue.depthStencil.stencil;
      }

      if (clear_att[clear_count].aspectMask != 0)
         clear_count++;

      if (clear_count > 0) {
         const VkClearRect clear_rect = {
            .rect = render->area,
            .baseArrayLayer = 0,
            .layerCount = render->view_mask ? 1 : render->layer_count,
         };

         hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                                clear_att, 1, &clear_rect);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   hk_cmd_buffer_end_graphics(cmd);

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         need_resolve = true;

      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));

   if (need_resolve) {
      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}

void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const gl_shader_stage stage,
                            struct hk_shader *shader)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   /* When a pipeline with tess shaders is bound we need to re-upload the
    * tessellation parameters at flush_ts_state, as the domain origin can be
    * dynamic.
    */
   if (stage == MESA_SHADER_TESS_EVAL)
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_TS_DOMAIN_ORIGIN);

   /* Emitting SET_HYBRID_ANTI_ALIAS_CONTROL requires the fragment shader */
   if (stage == MESA_SHADER_FRAGMENT) {
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

static uint32_t
hk_pipeline_bind_group(gl_shader_stage stage)
{
   return stage;
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.shaders_dirty == 0)
      return;

   /* Map shader types to shaders */
   struct hk_shader *type_shader[6] = {
      NULL,
   };
   uint32_t types_dirty = 0;

   const uint32_t gfx_stages =
      BITFIELD_BIT(MESA_SHADER_VERTEX) | BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
      BITFIELD_BIT(MESA_SHADER_TESS_EVAL) | BITFIELD_BIT(MESA_SHADER_GEOMETRY) |
      BITFIELD_BIT(MESA_SHADER_FRAGMENT);

   u_foreach_bit(stage, cmd->state.gfx.shaders_dirty & gfx_stages) {
      /* TODO: compact? */
      uint32_t type = stage;
      types_dirty |= BITFIELD_BIT(type);

      /* Only copy non-NULL shaders because mesh/task alias with vertex and
       * tessellation stages.
       */
      if (cmd->state.gfx.shaders[stage] != NULL) {
         assert(type < ARRAY_SIZE(type_shader));
         assert(type_shader[type] == NULL);
         type_shader[type] = cmd->state.gfx.shaders[stage];
      }
   }

   u_foreach_bit(type, types_dirty) {
      struct hk_shader *shader = type_shader[type];

      /* We always map index == type */
      // const uint32_t idx = type;

      if (shader == NULL)
         continue;

      /* TODO */
   }

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   /* TODO: gs/tes */

   /* If we have a new VS/FS pair, UVS locations may have changed so need to
    * relink. We do this here because there's no dependence on the fast linked
    * shaders.
    */
   agx_assign_uvs(&gfx->linked_varyings, &vs->info.uvs,
                  fs ? fs->info.fs.interp.flat : 0,
                  fs ? fs->info.fs.interp.linear : 0);

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   desc->root_dirty = true;

   for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
      desc->root.draw.uvs_index[i] = gfx->linked_varyings.slots[i];
   }
}

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader, dev->dev.libagx);

   struct agx_shader_key backend_key = {
      .needs_g13x_coherency = (dev->dev.params.gpu_generation == 13 &&
                               dev->dev.params.num_clusters_total > 1) ||
                              dev->dev.params.num_dies > 1,
      .libagx = dev->dev.libagx,
      .secondary = true,
      .no_stop = !stop,
   };

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);

   agx_compile_shader_nir(b.shader, &backend_key, NULL, part);

   ralloc_free(b.shader);

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return part;
}

static struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part = hk_get_prolog_epilog_locked(
      dev, key, builder, preprocess_nir, stop, cf_base);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           agx_nir_vs_prolog, false, false, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    agx_nir_fs_prolog, false, false,
                                    key->prolog.cf_base);
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           agx_nir_fs_epilog, true, epilog_stop, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent)
      linked = ent->data;
   else if (shader->info.stage == MESA_SHADER_VERTEX)
      linked = hk_get_fast_linked_locked_vs(dev, shader, key);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      linked = hk_get_fast_linked_locked_fs(dev, shader, key);
   else
      unreachable("invalid stage");

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static void
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   gl_shader_stage stage = shader->info.stage;

   if (cmd->state.gfx.linked[stage] != new) {
      cmd->state.gfx.linked[stage] = new;
      cmd->state.gfx.linked_dirty |= BITFIELD_BIT(stage);
   }
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   if (enabled) {
      agx_ppp_push(ppp, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_ppp_push(ppp, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.viewport_count, 1);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    *
    * which is software-visible and can cause faults with
    * eMRT when the framebuffer is not a multiple of the tile size.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < dyn->vp.scissor_count; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   struct agx_scissor_packed *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_pack(scissors + i, SCISSOR, cfg) {
         cfg.min_x = minx[i];
         cfg.min_y = miny[i];
         cfg.max_x = maxx[i];
         cfg.max_y = maxy[i];

         /* These settings in conjunction with the PPP control depth clip/clamp
          * settings implement depth clip/clamping. Properly setting them
          * together is required for conformant depth clip enable.
          *
          * TODO: Reverse-engineer the finer interactions here.
          */
         if (dyn->rs.depth_clamp_enable) {
            cfg.min_z = MIN2(vp->minDepth, vp->maxDepth);
            cfg.max_z = MAX2(vp->minDepth, vp->maxDepth);
         } else {
            cfg.min_z = 0.0;
            cfg.max_z = 1.0;
         }
      }
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_alloc(cmd, size, 64);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   agx_ppp_fini(out, &ppp);
}

static enum agx_object_type
translate_object_type(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;

   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return AGX_OBJECT_TYPE_LINE;

   default:
      return AGX_OBJECT_TYPE_TRIANGLE;
   }
}

static uint32_t
translate_primitive_topology(VkPrimitiveTopology prim)
{
   switch (prim) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return AGX_PRIMITIVE_POINTS;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return AGX_PRIMITIVE_LINES;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
   case VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA:
#pragma GCC diagnostic pop
      return AGX_PRIMITIVE_TRIANGLES;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;

   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      unreachable("todo: geometry shaders");

   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
      unreachable("todo: tessellation");

   default:
      unreachable("Invalid primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   uint8_t *out = cs->current;
   agx_push(out, VDM_STATE, cfg) {
      cfg.restart_index_present = true;
   }

   agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
      cfg.value = cmd->state.gfx.index.restart;
   }

   cs->current = out;
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      unreachable("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   bool vs_dirty = IS_SHADER_DIRTY(VERTEX);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull =
         IS_DIRTY(RS_CULL_MODE) || IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
         IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
         IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
         IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || (gfx->dirty & HK_DIRTY_PROVOKING),
      .cull_2 = varyings_dirty,

      /* With a null FS, the fragment shader PPP word is ignored and doesn't
       * need to be present.
       */
      .fragment_shader = fs && (fs_dirty || linked_fs_dirty || varyings_dirty ||
                                gfx->descriptors.root_dirty),

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_alloc(cmd, size, 64);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      if (linked_fs) {
         /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
          * main fragment control word and has to be combined into the secondary
          * word for reliable behaviour.
          */
         agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                             linked_fs->b.fragment_control) {

            cfg.tag_write_disable = dyn->rs.rasterizer_discard_enable;
         }
      } else {
         /* If there is no fragment shader, we must disable tag writes to avoid
          * executing the missing shader. This optimizes depth-only passes.
          */
         agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
            cfg.tag_write_disable = true;
            cfg.pass_type = AGX_PASS_TYPE_OPAQUE;
         }
      }
   }

   struct agx_fragment_face_packed fragment_face;
   struct agx_fragment_face_2_packed fragment_face_2;

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;

         /* TODO: flip the default? */
         if (fs)
            cfg.conservative_depth = 0;
      }

      if (fs)
         agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = vs->info.uvs.osel;

      if (linked_fs) {
         agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                   &linked_fs->b.osel);
      } else {
         agx_ppp_push_packed(&ppp, &osel, OUTPUT_SELECT);
      }
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;
         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = dyn->rs.rasterizer_discard_enable;

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, fs, linked_fs);
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = vs->info.uvs.size;
      }
   }

   agx_ppp_fini(out, &ppp);
}

static void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, uint32_t *draw_params,
                       uint64_t draw_params_gpu)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];

   if (!vk_dynamic_graphics_state_any_dirty(dyn) &&
       !(gfx->dirty & ~HK_DIRTY_INDEX) && !gfx->descriptors.root_dirty &&
       !gfx->shaders_dirty && !vs->b.info.uses_draw_id &&
       !vs->b.info.uses_base_param &&
       !(gfx->linked[MESA_SHADER_VERTEX] &&
         gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param))
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;

   struct hk_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   /* TODO: gs/tes */

   bool vs_dirty = IS_SHADER_DIRTY(VERTEX);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   if (IS_DIRTY(CB_BLEND_CONSTANTS)) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants) &&
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      desc->root.draw.api_sample_mask = dyn->ms.sample_mask;
      desc->root_dirty = true;
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      desc->root.draw.force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      desc->root_dirty = true;
   }

   /* The main shader must not run tests if the epilog will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   bool epilog_discards = dyn->ms.alpha_to_coverage_enable ||
                          (fs && (fs->info.fs.epilog_key.write_z ||
                                  fs->info.fs.epilog_key.write_s));
   epilog_discards &= !nontrivial_force_early;

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE)) {
      desc->root.draw.no_epilog_discard = !epilog_discards ? ~0 : 0;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || IS_SHADER_DIRTY(VERTEX)) {

      struct hk_fast_link_key_vs key = {
         /* TODO */
         .prolog.hw = true,
      };

      static_assert(sizeof(key.prolog.component_mask) ==
                    sizeof(vs->info.vs.attrib_components_read));
      BITSET_COPY(key.prolog.component_mask,
                  vs->info.vs.attrib_components_read);

      u_foreach_bit(a, dyn->vi->attributes_valid) {
         struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];

         assert(dyn->vi->bindings_valid & BITFIELD_BIT(attr.binding));
         struct vk_vertex_binding_state binding =
            dyn->vi->bindings[attr.binding];

         /* nir_assign_io_var_locations compacts vertex inputs, eliminating
          * unused inputs. We need to do the same here to match the locations.
          */
         unsigned slot =
            util_bitcount64(vs->info.vs.attribs_read & BITFIELD_MASK(a));

         key.prolog.attribs[slot] = (struct agx_velem_key){
            .format = vk_format_to_pipe_format(attr.format),
            .stride = dyn->vi_binding_strides[attr.binding],
            .divisor = binding.divisor,
            .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
         };
      }

      hk_update_fast_linked(cmd, vs, &key);
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) || IS_SHADER_DIRTY(VERTEX) ||
       (gfx->dirty & HK_DIRTY_VB)) {

      u_foreach_bit(a, dyn->vi->attributes_valid) {
         struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];

         unsigned slot =
            util_bitcount64(vs->info.vs.attribs_read & BITFIELD_MASK(a));

         desc->root_dirty = true;
         desc->root.draw.attrib_base[slot] =
            gfx->vb[attr.binding].addr + attr.offset;
      }
   }

   if (IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS)) {

      if (fs) {
         unsigned samples_shaded = 0;
         if (fs->info.fs.epilog_key.sample_shading)
            samples_shaded = dyn->ms.rasterization_samples;

         unsigned tib_sample_mask =
            BITFIELD_MASK(dyn->ms.rasterization_samples);
         unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
         bool has_sample_mask = api_sample_mask != tib_sample_mask;

         struct hk_fast_link_key_fs key = {
            .prolog.statistics = false /* TODO */,
            .prolog.cull_distance_size = vs->info.vs.cull_distance_array_size,
            .prolog.api_sample_mask = has_sample_mask ? api_sample_mask : 0xff,
            .nr_samples_shaded = samples_shaded,
         };

         bool prolog_discards =
            has_sample_mask || key.prolog.cull_distance_size;

         bool needs_prolog = key.prolog.statistics || prolog_discards;

         if (needs_prolog) {
            /* With late main shader tests, the prolog runs tests if neither the
             * main shader nor epilog will.
             *
             * With (nontrivial) early main shader tests, the prolog does not
             * run tests, the tests will run at the start of the main shader.
             * This ensures tests are after API sample mask and cull distance
             * discards.
             */
            key.prolog.run_zs_tests = !nontrivial_force_early &&
                                      !fs->b.info.writes_sample_mask &&
                                      !epilog_discards && prolog_discards;

            if (key.prolog.cull_distance_size) {
               key.prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
            }
         }

         key.epilog = (struct agx_fs_epilog_key){
            .link = fs->info.fs.epilog_key,
            .nr_samples = MAX2(dyn->ms.rasterization_samples, 1),
            .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
            .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
            .blend.logicop_func = dyn->cb.logic_op_enable
                                     ? vk_logic_op_to_pipe(dyn->cb.logic_op)
                                     : PIPE_LOGICOP_COPY,
         };

         key.epilog.link.already_ran_zs |= nontrivial_force_early;

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            key.epilog.rt_formats[i] =
               vk_format_to_pipe_format(render->color_att[i].vk_format);

            const struct vk_color_blend_attachment_state *cb =
               &dyn->cb.attachments[i];

            bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
            unsigned write_mask = write_enable ? cb->write_mask : 0;

            /* nir_lower_blend always blends, so use a default blend state when
             * blending is disabled at an API level.
             */
            if (!dyn->cb.attachments[i].blend_enable) {
               key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
                  .colormask = write_mask,
                  .rgb_func = PIPE_BLEND_ADD,
                  .alpha_func = PIPE_BLEND_ADD,
                  .rgb_src_factor = PIPE_BLENDFACTOR_ONE,
                  .alpha_src_factor = PIPE_BLENDFACTOR_ONE,
                  .rgb_dst_factor = PIPE_BLENDFACTOR_ZERO,
                  .alpha_dst_factor = PIPE_BLENDFACTOR_ZERO,
               };
            } else {
               key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
                  .colormask = write_mask,

                  .rgb_src_factor =
                     vk_blend_factor_to_pipe(cb->src_color_blend_factor),

                  .rgb_dst_factor =
                     vk_blend_factor_to_pipe(cb->dst_color_blend_factor),

                  .rgb_func = vk_blend_op_to_pipe(cb->color_blend_op),

                  .alpha_src_factor =
                     vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),

                  .alpha_dst_factor =
                     vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor),

                  .alpha_func = vk_blend_op_to_pipe(cb->alpha_blend_op),
               };
            }
         }

         hk_update_fast_linked(cmd, fs, &key);
      } else {
         /* TODO: prolog without fs needs to work too... */
         if (cmd->state.gfx.linked[MESA_SHADER_FRAGMENT] != NULL) {
            cmd->state.gfx.linked_dirty |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);
            cmd->state.gfx.linked[MESA_SHADER_FRAGMENT] = NULL;
         }
      }
   }

   /* If the vertex shader uses draw parameters, vertex uniforms are dirty every
    * draw. Fragment uniforms are unaffected.
    *
    * For a direct draw, we upload the draw parameters as-if indirect to
    * avoid keying to indirectness.
    */
   if (gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (draw_params) {
         assert(draw_params_gpu == 0);
         draw_params_gpu = hk_pool_upload(cmd, draw_params, 8, 8);
      }

      gfx->draw_params = draw_params_gpu;
   } else {
      gfx->draw_params = 0;
   }

   if (gfx->shaders[MESA_SHADER_VERTEX]->b.info.uses_draw_id) {
      /* TODO: rodata? */
      gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, 2, 4);
   } else {
      gfx->draw_id_ptr = 0;
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY)) {
      gfx->topology = translate_primitive_topology(dyn->ia.primitive_topology);
      gfx->object_type = translate_object_type(dyn->ia.primitive_topology);
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX)) {
      unsigned provoking;
      if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
         provoking = 2;
      else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
         provoking = 1;
      else
         provoking = 0;

      if (provoking != gfx->provoking) {
         gfx->provoking = provoking;
         gfx->dirty |= HK_DIRTY_PROVOKING;

         bool first_fan = (provoking == 1);

         if (first_fan != gfx->descriptors.root.draw.first_fan) {
            gfx->descriptors.root.draw.first_fan = first_fan;
            gfx->descriptors.root_dirty = true;
         }
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    */
   if (dyn->ms.rasterization_samples &&
       gfx->render.tilebuffer.nr_samples != dyn->ms.rasterization_samples) {

      assert(gfx->render.tilebuffer.nr_samples == 0);

      unsigned nr_samples = MAX2(dyn->ms.rasterization_samples, 1);
      gfx->render.tilebuffer.nr_samples = nr_samples;
      agx_tilebuffer_pack_usc(&gfx->render.tilebuffer);
      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint32_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if (!cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      gfx->descriptors.root.draw.ppp_multisamplectl = ctrl;
      gfx->descriptors.root_dirty = true;
   }

   /* Root must be uploaded after the above, which touch the root */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   }

   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   if ((gfx->dirty & HK_DIRTY_PROVOKING) || vs_dirty || linked_fs_dirty) {
      unsigned bindings = linked_fs ? linked_fs->b.cf.nr_bindings : 0;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         struct agx_ptr t = hk_pool_usc_alloc(cmd, linkage_size, 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = t.gpu;
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if ((gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
       IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty ||
       gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, vs);

      agx_push(out, VDM_STATE, cfg) {
         cfg.vertex_shader_word_0_present = true;
         cfg.vertex_shader_word_1_present = true;
         cfg.vertex_outputs_present = true;
         cfg.vertex_unknown_present = true;
      }

      agx_push_packed(out, vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

      agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, vs, linked_vs);
      }

      agx_push_packed(out, vs->info.uvs.vdm, VDM_STATE_VERTEX_OUTPUTS);

      agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
         cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
         cfg.unknown_4 = cfg.unknown_5 = dyn->rs.rasterizer_discard_enable;
         cfg.generate_primitive_id = gfx->generate_primitive_id;
      }

      /* Pad up to a multiple of 8 bytes */
      memset(out, 0, 4);
      out += 4;
   }

   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, AGX_DEPTH_BIAS_LENGTH);

      agx_pack(ptr, DEPTH_BIAS, cfg) {
         cfg.depth_bias = dyn->rs.depth_bias.constant;
         cfg.slope_scale = dyn->rs.depth_bias.slope;
         cfg.clamp = dyn->rs.depth_bias.clamp;

         /* Value from the PowerVR driver. */
         if (render->depth_att.vk_format == VK_FORMAT_D16_UNORM) {
            cfg.depth_bias /= (1 << 15);
         }
      }
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      hk_flush_vp_state(cmd, cs, &out);
   }

   hk_flush_ppp_state(cmd, cs, &out);
   cs->current = out;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;
}

static struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   uint32_t *draw_params, uint64_t draw_params_gpu)
{
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return NULL;

   /* XXX: should stream link instead for performance! */
   if (cs->current + 0x2000 > cs->end) {
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
      if (!cs)
         return NULL;
      // printf("Splitting graphics batch due to missing stream links\n");
   }

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   hk_flush_shaders(cmd);

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   if ((gfx->dirty & HK_DIRTY_INDEX) && cmd->state.gfx.index.restart)
      hk_flush_index(cmd, cs);

   hk_flush_dynamic_state(cmd, cs, draw_id, draw_params, draw_params_gpu);
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

   /* TODO: check if necessary, blob does this */
   cmd->state.gfx.index.buffer.range =
      align(cmd->state.gfx.index.buffer.range, 4);

   cmd->state.gfx.dirty |= HK_DIRTY_INDEX;
}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

static void
hk_draw(VkCommandBuffer commandBuffer, uint16_t draw_id, uint32_t vertexCount,
        uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   uint32_t params[] = {firstVertex, firstInstance};

   hk_foreach_view(cmd) {
      struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, params, 0);
      if (!cs)
         return;

      assert(cs->current + 0x1000 < cs->end);

      uint8_t *out = cs->current;

      agx_push(out, INDEX_LIST, cfg) {
         cfg.primitive = cmd->state.gfx.topology;
         cfg.instance_count_present = true;
         cfg.index_count_present = true;
         cfg.start_present = true;
      }

      agx_push(out, INDEX_LIST_COUNT, cfg) {
         cfg.count = vertexCount;
      }

      agx_push(out, INDEX_LIST_INSTANCES, cfg) {
         cfg.count = instanceCount;
      }

      agx_push(out, INDEX_LIST_START, cfg) {
         cfg.start = firstVertex;
      }

      cs->current = out;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   hk_draw(commandBuffer, 0, vertexCount, instanceCount, firstVertex,
           firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      hk_draw(commandBuffer, i, pVertexInfo->vertexCount, instanceCount,
              pVertexInfo->firstVertex, firstInstance);

      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   uint32_t params[] = {vertexOffset, firstInstance};

   hk_foreach_view(cmd) {
      struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, params, 0);
      if (!cs)
         return;

      assert(cs->current + 0x1000 < cs->end);

      uint8_t *out = cs->current;

      enum agx_index_size index_shift = cmd->state.gfx.index.size;
      uint64_t ib = cmd->state.gfx.index.buffer.addr;
      ib += (firstIndex << index_shift);

      agx_push(out, INDEX_LIST, cfg) {
         cfg.primitive = cmd->state.gfx.topology;
         cfg.instance_count_present = true;
         cfg.index_count_present = true;
         cfg.start_present = true;

         cfg.restart_enable = dyn->ia.primitive_restart_enable;
         cfg.index_buffer_hi = ib >> 32;
         cfg.index_size = cmd->state.gfx.index.size;
         cfg.index_buffer_present = true;
         cfg.index_buffer_size_present = true;
      }

      agx_push(out, INDEX_LIST_BUFFER_LO, cfg) {
         cfg.buffer_lo = ib;
      }

      agx_push(out, INDEX_LIST_COUNT, cfg) {
         cfg.count = indexCount;
      }

      agx_push(out, INDEX_LIST_INSTANCES, cfg) {
         cfg.count = instanceCount;
      }

      agx_push(out, INDEX_LIST_START, cfg) {
         cfg.start = vertexOffset;
      }

      agx_push(out, INDEX_LIST_BUFFER_SIZE, cfg) {
         cfg.size = cmd->state.gfx.index.buffer.range;
      }

      cs->current = out;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const uint32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      hk_foreach_view(cmd) {
         uint64_t addr = hk_buffer_address(buffer, offset) + stride * draw_id;

         /* Grab the firstVertex/baseInstance vec2 */
         uint64_t params = addr + offsetof(VkDrawIndirectCommand, firstVertex);
         struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, NULL, params);
         if (!cs)
            return;

         assert(cs->current + 0x1000 < cs->end);

         uint8_t *out = cs->current;

         agx_push(out, INDEX_LIST, cfg) {
            cfg.primitive = cmd->state.gfx.topology;
            cfg.indirect_buffer_present = true;
         }

         agx_push(out, INDEX_LIST_INDIRECT_BUFFER, cfg) {
            cfg.address_hi = addr >> 32;
            cfg.address_lo = addr;
         }

         cs->current = out;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      hk_foreach_view(cmd) {
         uint64_t addr = hk_buffer_address(buffer, offset) + stride * draw_id;

         /* Grab the firstVertex/baseInstance vec2 */
         uint64_t draw_params =
            addr + offsetof(VkDrawIndexedIndirectCommand, vertexOffset);

         struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, NULL, draw_params);
         if (!cs)
            return;

         assert(cs->current + 0x1000 < cs->end);

         uint8_t *out = cs->current;
         uint64_t ib = cmd->state.gfx.index.buffer.addr;

         agx_push(out, INDEX_LIST, cfg) {
            cfg.primitive = cmd->state.gfx.topology;
            cfg.indirect_buffer_present = true;
            cfg.restart_enable = dyn->ia.primitive_restart_enable;
            cfg.index_buffer_hi = ib >> 32;
            cfg.index_size = cmd->state.gfx.index.size;
            cfg.index_buffer_present = true;
            cfg.index_buffer_size_present = true;
         }

         agx_push(out, INDEX_LIST_BUFFER_LO, cfg) {
            cfg.buffer_lo = ib;
         }

         agx_push(out, INDEX_LIST_INDIRECT_BUFFER, cfg) {
            cfg.address_hi = addr >> 32;
            cfg.address_lo = addr & BITFIELD_MASK(32);
         }

         agx_push(out, INDEX_LIST_BUFFER_SIZE, cfg) {
            cfg.size = cmd->state.gfx.index.buffer.range;
         }

         cs->current = out;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   unreachable("TODO");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   unreachable("TODO");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   unreachable("TODO");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   unreachable("TODO");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   unreachable("stub");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   unreachable("stub");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   unreachable("stub");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   unreachable("stub");
}
