// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/disjoint.mlir \
// RUN:   | FileCheck %s --check-prefix=DISJOINT
// RUN: not loom-raise-opt --loom-lower-scf-to-dfg %t.dir/overlap.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=OVERLAP

// A 2x2 scf.parallel nest with constant bounds is lowered into four lanes.
// Each lane loads, increments and stores target[2 * iv0 + iv1]. The four
// addresses are distinct constants, so the lanes are independent: every
// lane's load takes the graph start, each store waits only for its own load,
// and the four stores are joined once at the return.

// DISJOINT-LABEL: dataflow.graph private @g_thread_0
// DISJOINT-SAME:  %[[START:[^:]+]]: none, %[[MEM:[^:]+]]: memref<?xindex>
// DISJOINT-DAG:   %[[C1:.*]] = dataflow.constant %[[START]] {const_value = 1 : index} : index
// DISJOINT-DAG:   %[[C2:.*]] = dataflow.constant %[[START]] {const_value = 2 : index} : index
// DISJOINT-DAG:   %[[C0:.*]] = dataflow.constant %[[START]] {const_value = 0 : index} : index
// DISJOINT:       %[[M0:.*]] = arith.muli %[[C0]], %[[C2]] : index
// DISJOINT-NEXT:  %[[A0:.*]] = arith.addi %[[M0]], %[[C0]] : index
// DISJOINT:       %[[M1:.*]] = arith.muli %[[C0]], %[[C2]] : index
// DISJOINT-NEXT:  %[[A1:.*]] = arith.addi %[[M1]], %[[C1]] : index
// DISJOINT:       %[[M2:.*]] = arith.muli %[[C1]], %[[C2]] : index
// DISJOINT-NEXT:  %[[A2:.*]] = arith.addi %[[M2]], %[[C0]] : index
// DISJOINT:       %[[M3:.*]] = arith.muli %[[C1]], %[[C2]] : index
// DISJOINT-NEXT:  %[[A3:.*]] = arith.addi %[[M3]], %[[C1]] : index
// DISJOINT:       %{{.*}}, %[[K0:.*]] = dataflow.load %[[MEM]][%[[A0]]] %[[START]] : memref<?xindex>
// DISJOINT-NEXT:  %[[S0:.*]] = dataflow.store %[[MEM]][%[[A0]]] %{{.*}} %[[K0]] : memref<?xindex>
// DISJOINT-NEXT:  %{{.*}}, %[[K1:.*]] = dataflow.load %[[MEM]][%[[A1]]] %[[START]] : memref<?xindex>
// DISJOINT-NEXT:  %[[S1:.*]] = dataflow.store %[[MEM]][%[[A1]]] %{{.*}} %[[K1]] : memref<?xindex>
// DISJOINT-NEXT:  %{{.*}}, %[[K2:.*]] = dataflow.load %[[MEM]][%[[A2]]] %[[START]] : memref<?xindex>
// DISJOINT-NEXT:  %[[S2:.*]] = dataflow.store %[[MEM]][%[[A2]]] %{{.*}} %[[K2]] : memref<?xindex>
// DISJOINT-NEXT:  %{{.*}}, %[[K3:.*]] = dataflow.load %[[MEM]][%[[A3]]] %[[START]] : memref<?xindex>
// DISJOINT-NEXT:  %[[S3:.*]] = dataflow.store %[[MEM]][%[[A3]]] %{{.*}} %[[K3]] : memref<?xindex>
// DISJOINT-NEXT:  %[[RET:.*]]:4 = dataflow.sync %[[S0]], %[[S1]], %[[S2]], %[[S3]]
// DISJOINT-NEXT:  dataflow.graph.return %[[RET]]#0 : none
// DISJOINT-NOT:   scf.parallel

// The same nest with target[iv0]: the two inner lanes of each outer lane
// load and store one address, so the lanes are not independent and the
// lowering fails.

// OVERLAP: error: loom-lower-graph-memory: parallel lanes have overlapping plain memory effects

//--- disjoint.mlir
dataflow.thread private @thread_0 domain(#dataflow.thread_domain<dense>)(
    %memory: memref<?xindex>) ctrl (%ctrl: none) {
  "loom.spatial_region"(%memory)
      <{operandSegmentSizes = array<i32: 0, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%target: memref<?xindex>):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      scf.parallel (%iv0) = (%zero) to (%two) step (%one) {
        scf.parallel (%iv1) = (%zero) to (%two) step (%one) {
          %mx1 = arith.muli %iv0, %two : index
          %ix1 = arith.addi %mx1, %iv1 : index
          %ld = memref.load %target[%ix1] : memref<?xindex>
          %ad = arith.addi %ld, %one : index
          memref.store %ad, %target[%ix1] : memref<?xindex>
          scf.reduce
        }
        scf.reduce
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "g_thread_0", source_maps = []} :
      (memref<?xindex>) -> ()
  dataflow.thread.yield
}

//--- overlap.mlir
dataflow.thread private @thread_0 domain(#dataflow.thread_domain<dense>)(
    %memory: memref<?xindex>) ctrl (%ctrl: none) {
  "loom.spatial_region"(%memory)
      <{operandSegmentSizes = array<i32: 0, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%target: memref<?xindex>):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      scf.parallel (%iv0) = (%zero) to (%two) step (%one) {
        scf.parallel (%iv1) = (%zero) to (%two) step (%one) {
          %ld = memref.load %target[%iv0] : memref<?xindex>
          %ad = arith.addi %ld, %one : index
          memref.store %ad, %target[%iv0] : memref<?xindex>
          scf.reduce
        }
        scf.reduce
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "g_thread_0", source_maps = []} :
      (memref<?xindex>) -> ()
  dataflow.thread.yield
}
