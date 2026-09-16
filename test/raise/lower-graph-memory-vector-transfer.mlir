// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// Vector memory actors inside a construction-local graph: a masked gather and
// scatter that already carry a vector address, and vector.transfer_read /
// vector.transfer_write pairs under scf.if and scf.for, with and without a
// mask. Each transfer becomes one addressed dataflow.load / dataflow.store
// firing that keeps its mask, and the frontier equations order the pairs.

// CHECK-LABEL: dataflow.graph private @vector_transfers
// CHECK-SAME:  %[[START:[^,]+]]: none, %[[I:[^,]+]]: index, %[[C:[^,]+]]: i1, %[[M:[^,]+]]: vector<4xi1>, %[[AV:[^,]+]]: vector<4xindex>, %[[A:[^,]+]]: memref<16xi32>, %[[B:[^,]+]]: memref<16xi32>

// The gather/scatter pair keeps its vector address and mask, and the scatter
// waits for the gather.
// CHECK:      %[[G:[^,]+]], %[[GD:[^ ]+]] = dataflow.load %[[A]][%[[AV]]] %[[START]] mask %[[M]] : memref<16xi32>, vector<4xindex>, vector<4xi32>
// CHECK-NEXT: dataflow.store %[[B]][%[[AV]]] %[[G]] %[[GD]] mask %[[M]] : memref<16xi32>, vector<4xindex>, vector<4xi32>

// The unmasked transfer pair under scf.if lowers to whole-vector actors.
// CHECK:      %[[R1:[^,]+]], %[[R1D:[^ ]+]] = dataflow.load %[[B]][{{.*}}] {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK:      dataflow.store %[[A]][{{.*}}] %[[R1]] {{.*}} : memref<16xi32>, vector<4xi32>

// The masked transfer pair under scf.for keeps the mask through the loop.
// CHECK:      dataflow.stream
// CHECK:      %[[R2:[^,]+]], %[[R2D:[^ ]+]] = dataflow.load %[[A]][{{.*}}] {{.*}} mask %{{.*}} : memref<16xi32>, vector<4xi32>
// CHECK:      dataflow.store %[[B]][{{.*}}] %[[R2]] {{.*}} mask %{{.*}} : memref<16xi32>, vector<4xi32>

// The masked transfer pair under scf.if.
// CHECK:      %[[R3:[^,]+]], %[[R3D:[^ ]+]] = dataflow.load %[[B]][{{.*}}] {{.*}} mask %{{.*}} : memref<16xi32>, vector<4xi32>
// CHECK:      dataflow.store %[[A]][{{.*}}] %[[R3]] {{.*}} mask %{{.*}} : memref<16xi32>, vector<4xi32>
// CHECK-NOT:  vector.transfer_read
// CHECK-NOT:  vector.transfer_write
dataflow.graph private @vector_transfers(
    %start: none, %i: index, %c: i1, %m: vector<4xi1>, %av: vector<4xindex>,
    %a: memref<16xi32>, %b: memref<16xi32>) -> ()
    attributes {input_segments = array<i32: 4, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %pad = arith.constant 0 : i32
  %g, %gd = dataflow.load %a[%av] %start mask %m : memref<16xi32>, vector<4xindex>, vector<4xi32>
  %sd = dataflow.store %b[%av] %g %start mask %m : memref<16xi32>, vector<4xindex>, vector<4xi32>
  scf.if %c {
    %v1 = vector.transfer_read %b[%i], %pad {in_bounds = [true]} : memref<16xi32>, vector<4xi32>
    vector.transfer_write %v1, %a[%i] {in_bounds = [true]} : vector<4xi32>, memref<16xi32>
  }
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %k = %lb to %ub step %step {
    %v2 = vector.transfer_read %a[%i], %pad, %m {in_bounds = [true]} : memref<16xi32>, vector<4xi32>
    vector.transfer_write %v2, %b[%i], %m {in_bounds = [true]} : vector<4xi32>, memref<16xi32>
  }
  scf.if %c {
    %v3 = vector.transfer_read %b[%i], %pad, %m {in_bounds = [true]} : memref<16xi32>, vector<4xi32>
    vector.transfer_write %v3, %a[%i], %m {in_bounds = [true]} : vector<4xi32>, memref<16xi32>
  }
  dataflow.graph.return %start : none
}
