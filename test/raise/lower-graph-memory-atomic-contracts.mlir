// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory \
// RUN:   %t.dir/accepted.mlir | FileCheck %s
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/unaligned.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNALIGNED
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/target-scope.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=TARGET-SCOPE

// An atomic access whose alignment is not a power of two, and one whose
// synchronization scope is target-specific, fail closed.
// UNALIGNED: error: loom-lower-graph-memory: atomic source requires a supported ordering/scope and an explicit power-of-two alignment
// TARGET-SCOPE: error: loom-lower-graph-memory: atomic source requires a supported ordering/scope and an explicit power-of-two alignment

//--- accepted.mlir
// LLVM atomic and volatile accesses inside a construction-local graph are
// normalized into dataflow memory actors that carry the source contract:
// ordering, synchronization scope, alignment and RMW kind. Every access in a
// graph uses one address, so from the first read-modify-write on, each actor
// takes the previous actor's done as its control.

// CHECK-LABEL: dataflow.graph private @atomic_sequence
// CHECK:      %[[V:[^,]+]], %[[VD:[^ ]+]] = dataflow.load %[[MEM:arg[0-9]+]][%[[P:[0-9#]+]]] %{{.*}} {contract = #dataflow.atomic_access<ordering = seq_cst, sync_scope = <system>, source_alignment_bytes = 4>} : memref<?xi32>, !llvm.ptr
// CHECK:      %[[X:[^,]+]], %[[XD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %{{.*}} {contract = #dataflow.rmw_contract<kind = xchg, access = <ordering = acquire, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %[[U:[^,]+]], %[[UD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[XD]] {contract = #dataflow.rmw_contract<kind = umax, access = <ordering = seq_cst, sync_scope = <single_thread>, source_alignment_bytes = 8>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %[[F:[^ ]+]] = dataflow.fence %[[UD]] {contract = #dataflow.fence_contract<ordering = release, sync_scope = <system>>}
// CHECK-NEXT: dataflow.store %[[MEM]][%[[P]]] %{{.*}} %[[F]] {contract = #dataflow.atomic_access<ordering = seq_cst, sync_scope = <system>, source_alignment_bytes = 4>} : memref<?xi32>, !llvm.ptr
// CHECK-NOT:  llvm.load
// CHECK-NOT:  llvm.atomicrmw
// CHECK-NOT:  llvm.fence
// CHECK-NOT:  llvm.store
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @atomic_sequence(
      %start: none, %base: !llvm.ptr, %desired: i32, %index: i64, %cond: i1) -> ()
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    scf.if %cond {
      %val = llvm.load %ptr atomic seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
      %rmw1 = llvm.atomicrmw xchg %ptr, %desired acquire {alignment = 4 : i64} : !llvm.ptr, i32
      %rmw2 = llvm.atomicrmw umax %ptr, %desired syncscope("singlethread") seq_cst {alignment = 8 : i64} : !llvm.ptr, i32
      llvm.fence syncscope("system") release
      llvm.store %desired, %ptr atomic seq_cst {alignment = 4 : i64} : i32, !llvm.ptr
    }
    dataflow.graph.return %start : none
  }
}

// -----

// The remaining integer RMW kinds and a volatile plain load.

// CHECK-LABEL: dataflow.graph private @rmw_kinds
// CHECK:      %{{.*}}, %[[MD:[^ ]+]] = dataflow.atomic_rmw %[[MEM:arg[0-9]+]][%[[P:[0-9#]+]]] %{{.*}} %{{.*}} {contract = #dataflow.rmw_contract<kind = umin, access = <ordering = acq_rel, sync_scope = <single_thread>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[XD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[MD]] {contract = #dataflow.rmw_contract<kind = xor, access = <ordering = acquire, sync_scope = <system>, source_alignment_bytes = 8>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[SD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[XD]] {contract = #dataflow.rmw_contract<kind = sub, access = <ordering = acquire, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[AD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[SD]] {contract = #dataflow.rmw_contract<kind = and, access = <ordering = monotonic, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[ND:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[AD]] {contract = #dataflow.rmw_contract<kind = nand, access = <ordering = monotonic, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[OD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[ND]] {contract = #dataflow.rmw_contract<kind = or, access = <ordering = monotonic, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[MXD:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[OD]] {contract = #dataflow.rmw_contract<kind = max, access = <ordering = monotonic, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: %{{.*}}, %[[MND:[^ ]+]] = dataflow.atomic_rmw %[[MEM]][%[[P]]] %{{.*}} %[[MXD]] {contract = #dataflow.rmw_contract<kind = min, access = <ordering = monotonic, sync_scope = <system>, source_alignment_bytes = 4>>} : memref<?xi32>, !llvm.ptr
// CHECK-NEXT: dataflow.load %[[MEM]][%[[P]]] %[[MND]] {contract = #dataflow.plain_access<is_volatile = true>} : memref<?xi32>, !llvm.ptr
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @rmw_kinds(
      %start: none, %base: !llvm.ptr, %desired: i32, %index: i64) -> ()
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %rmw0 = llvm.atomicrmw umin %ptr, %desired syncscope("singlethread") acq_rel {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw1 = llvm.atomicrmw _xor %ptr, %desired acquire {alignment = 8 : i64} : !llvm.ptr, i32
    %rmw2 = llvm.atomicrmw sub %ptr, %desired syncscope("system") acquire {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw3 = llvm.atomicrmw _and %ptr, %desired monotonic {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw4 = llvm.atomicrmw nand %ptr, %desired monotonic {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw5 = llvm.atomicrmw _or %ptr, %desired monotonic {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw6 = llvm.atomicrmw max %ptr, %desired monotonic {alignment = 4 : i64} : !llvm.ptr, i32
    %rmw7 = llvm.atomicrmw min %ptr, %desired monotonic {alignment = 4 : i64} : !llvm.ptr, i32
    %val = llvm.load volatile %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %start : none
  }
}

// -----

// A fence inside a loop, with the only address computation left dead: the
// fence keeps its contract, the loop is lowered, and the unused
// getelementptr is erased.

// CHECK-LABEL: dataflow.graph private @fence_in_loop
// CHECK-NOT:  llvm.getelementptr
// CHECK:      %[[IV:.*]], %[[PH:.*]] = dataflow.stream
// CHECK:      %[[TOK:.*]] = dataflow.carry %[[PH]], %arg0, %[[F:.*]] : none
// CHECK:      %[[D:.*]]:2 = dataflow.demux %[[PH]], %[[TOK]]
// CHECK:      %[[F]] = dataflow.fence %[[D]]#1 {contract = #dataflow.fence_contract<ordering = acquire, sync_scope = <system>>}
// CHECK-NOT:  llvm.fence
// CHECK-NOT:  llvm.getelementptr

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @fence_in_loop(
      %start: none, %base: !llvm.ptr, %index: i64) -> ()
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lower to %upper step %step {
      llvm.fence syncscope("system") acquire
    }
    dataflow.graph.return %start : none
  }
}

//--- unaligned.mlir
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @unaligned_atomic(
      %start: none, %base: !llvm.ptr, %index: i64) -> ()
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %val = llvm.load %ptr atomic seq_cst {alignment = 3 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %start : none
  }
}

//--- target-scope.mlir
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @target_scope_atomic(
      %start: none, %base: !llvm.ptr, %index: i64) -> ()
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %val = llvm.load %ptr atomic syncscope("agent") seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %start : none
  }
}
