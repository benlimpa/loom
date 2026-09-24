// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory \
// RUN:   %t.dir/accepted.mlir | FileCheck %s --implicit-check-not=memref.copy
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/subview.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=SUBVIEW

// memref.cast is the only accepted view form, so exporting a memref.subview
// of an incoming memory port is rejected.
// SUBVIEW: error: loom-lower-graph-memory: operation 'memref.subview' is not a registered canonical Dataflow actor or a supported graph-lowering operation

//--- accepted.mlir
// A graph exports a memref.cast view of an incoming memory port, and a
// graph-local allocation. Each export is the memref itself: no copy is made.

// CHECK-LABEL: dataflow.graph private @view_export
// CHECK:      %[[VIEW:.*]] = memref.cast %arg4 : memref<4xi32> to memref<?xi32>
// CHECK:      dataflow.store %[[VIEW]][{{.*}}] {{.*}} : memref<?xi32>
// CHECK:      dataflow.graph.return values() streams() memories(%[[VIEW]] : memref<?xi32>) complete(
// CHECK-NOT:  memref.store

dataflow.graph private @view_export(
    %start: none, %i: index, %c: i1, %v: i32, %m: memref<4xi32>) -> (memref<?xi32>)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %view = memref.cast %m : memref<4xi32> to memref<?xi32>
  scf.if %c {
    memref.store %v, %view[%i] : memref<?xi32>
  }
  dataflow.graph.return values() streams()
      memories(%view : memref<?xi32>) complete(%start : none)
}

// -----

// CHECK-LABEL: dataflow.graph private @fresh_export
// CHECK:      %[[SLOT:.*]] = memref.alloc() : memref<4xi32>
// CHECK:      dataflow.stream %arg1, %arg2, %arg3 step add while slt : i64
// CHECK:      %[[IDX:.*]] = arith.index_cast %iv : i64 to index
// CHECK:      dataflow.store %[[SLOT]][%[[IDX]]] {{.*}} : memref<4xi32>
// CHECK:      dataflow.graph.return values() streams() memories(%[[SLOT]] : memref<4xi32>) complete(
// CHECK-NOT:  scf.for

dataflow.graph private @fresh_export(
    %start: none, %lb: i64, %ub: i64, %step: i64, %v: i32) -> (memref<4xi32>)
    attributes {input_segments = array<i32: 4, 0, 0>,
                result_segments = array<i32: 0, 0, 1>} {
  %slot = memref.alloc() : memref<4xi32>
  scf.for %iv = %lb to %ub step %step : i64 {
    %idx = arith.index_cast %iv : i64 to index
    memref.store %v, %slot[%idx] : memref<4xi32>
  }
  dataflow.graph.return values() streams()
      memories(%slot : memref<4xi32>) complete(%start : none)
}

//--- subview.mlir
dataflow.graph private @subview_export(
    %start: none, %i: index, %c: i1, %v: i32, %m: memref<8xi32>) -> (memref<4xi32, strided<[1], offset: 2>>)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %view = memref.subview %m[2] [4] [1] : memref<8xi32> to memref<4xi32, strided<[1], offset: 2>>
  scf.if %c {
    memref.store %v, %view[%i] : memref<4xi32, strided<[1], offset: 2>>
  }
  dataflow.graph.return values() streams()
      memories(%view : memref<4xi32, strided<[1], offset: 2>>) complete(%start : none)
}
