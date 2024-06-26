/*
 * Copyright © 2022 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */
#include "helpers.h"

using namespace aco;

BEGIN_TEST(insert_waitcnt.ds_ordered_count)
   if (!setup_cs(NULL, GFX10_3))
      return;

   Operand def0(PhysReg(256), v1);
   Operand def1(PhysReg(257), v1);
   Operand def2(PhysReg(258), v1);
   Operand gds_base(PhysReg(259), v1);
   Operand chan_counter(PhysReg(260), v1);
   Operand m(m0, s1);

   Instruction* ds_instr;
   //>> ds_ordered_count %0:v[0], %0:v[3], %0:m0 offset0:3072 gds storage:gds semantics:volatile
   //! s_waitcnt lgkmcnt(0)
   ds_instr = bld.ds(aco_opcode::ds_ordered_count, def0, gds_base, m, 3072u, 0u, true);
   ds_instr->ds().sync = memory_sync_info(storage_gds, semantic_volatile);

   //! ds_add_rtn_u32 %0:v[1], %0:v[3], %0:v[4], %0:m0 gds storage:gds semantics:volatile,atomic,rmw
   ds_instr = bld.ds(aco_opcode::ds_add_rtn_u32, def1, gds_base, chan_counter, m, 0u, 0u, true);
   ds_instr->ds().sync = memory_sync_info(storage_gds, semantic_atomicrmw);

   //! s_waitcnt lgkmcnt(0)
   //! ds_ordered_count %0:v[2], %0:v[3], %0:m0 offset0:3840 gds storage:gds semantics:volatile
   ds_instr = bld.ds(aco_opcode::ds_ordered_count, def2, gds_base, m, 3840u, 0u, true);
   ds_instr->ds().sync = memory_sync_info(storage_gds, semantic_volatile);

   finish_waitcnt_test();
END_TEST

BEGIN_TEST(insert_waitcnt.clause)
   if (!setup_cs(NULL, GFX11))
      return;

   Definition def_v4(PhysReg(260), v1);
   Definition def_v5(PhysReg(261), v1);
   Definition def_v6(PhysReg(262), v1);
   Definition def_v7(PhysReg(263), v1);
   Operand op_v0(PhysReg(256), v1);
   Operand op_v4(PhysReg(260), v1);
   Operand op_v5(PhysReg(261), v1);
   Operand op_v6(PhysReg(262), v1);
   Operand op_v7(PhysReg(263), v1);
   Operand desc0(PhysReg(0), s4);

   //>> p_unit_test 0
   bld.pseudo(aco_opcode::p_unit_test, Operand::zero());

   //! v1: %0:v[4] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   //! v1: %0:v[5] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   //! v1: %0:v[6] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   //! v1: %0:v[7] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v4, desc0, op_v0, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v5, desc0, op_v0, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v6, desc0, op_v0, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v7, desc0, op_v0, Operand::zero(), 0, false);
   //! s_waitcnt vmcnt(0)
   //! v1: %0:v[4] = buffer_load_dword %0:s[0-3], %0:v[4], 0
   //! v1: %0:v[5] = buffer_load_dword %0:s[0-3], %0:v[5], 0
   //! v1: %0:v[6] = buffer_load_dword %0:s[0-3], %0:v[6], 0
   //! v1: %0:v[7] = buffer_load_dword %0:s[0-3], %0:v[7], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v4, desc0, op_v4, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v5, desc0, op_v5, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v6, desc0, op_v6, Operand::zero(), 0, false);
   bld.mubuf(aco_opcode::buffer_load_dword, def_v7, desc0, op_v7, Operand::zero(), 0, false);
   //! s_waitcnt vmcnt(0)
   //! buffer_store_dword %0:s[0-3], %0:v[0], 0, %0:v[4]
   //! buffer_store_dword %0:s[0-3], %0:v[0], 0, %0:v[5]
   //! buffer_store_dword %0:s[0-3], %0:v[0], 0, %0:v[6]
   //! buffer_store_dword %0:s[0-3], %0:v[0], 0, %0:v[7]
   bld.mubuf(aco_opcode::buffer_store_dword, desc0, op_v0, Operand::zero(), op_v4, 0, false);
   bld.mubuf(aco_opcode::buffer_store_dword, desc0, op_v0, Operand::zero(), op_v5, 0, false);
   bld.mubuf(aco_opcode::buffer_store_dword, desc0, op_v0, Operand::zero(), op_v6, 0, false);
   bld.mubuf(aco_opcode::buffer_store_dword, desc0, op_v0, Operand::zero(), op_v7, 0, false);

   //>> p_unit_test 1
   bld.reset(program->create_and_insert_block());
   bld.pseudo(aco_opcode::p_unit_test, Operand::c32(1));

   //! s4: %0:s[4-7] = s_load_dwordx4 %0:s[0-1], 0
   bld.smem(aco_opcode::s_load_dwordx4, Definition(PhysReg(4), s4), Operand(PhysReg(0), s2),
            Operand::zero());
   //! v1: %0:v[4] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v4, desc0, op_v0, Operand::zero(), 0, false);
   //! s_waitcnt vmcnt(0) lgkmcnt(0)
   //! v1: %0:v[5] = buffer_load_dword %0:s[4-7], %0:v[4], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v5, Operand(PhysReg(4), s4), op_v4, Operand::zero(),
             0, false);

   //>> p_unit_test 2
   bld.reset(program->create_and_insert_block());
   bld.pseudo(aco_opcode::p_unit_test, Operand::c32(2));

   //! v1: %0:v[4] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v4, desc0, op_v0, Operand::zero(), 0, false);
   //! v_nop
   bld.vop1(aco_opcode::v_nop);
   //! v1: %0:v[4] = buffer_load_dword %0:s[0-3], %0:v[0], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v4, desc0, op_v0, Operand::zero(), 0, false);
   //! s_waitcnt vmcnt(0)
   //! v1: %0:v[5] = buffer_load_dword %0:s[0-3], %0:v[4], 0
   bld.mubuf(aco_opcode::buffer_load_dword, def_v5, desc0, op_v4, Operand::zero(), 0, false);

   finish_waitcnt_test();
END_TEST
