// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// Tests use of [[ hc_waves_per_eu ]] CXX11-style attribute.
// Tests use of [[ hc_flat_workgroup_size ]] CXX11-style attribute.
// Tests use of [[ hc_max_workgroup_dim ]] CXX11-style attribute.

#include "hc.hpp"
#include "grid_launch.hpp"
#include "hc_am.hpp"
#include <iostream>

#define GRID_SIZE 16
#define TILE_SIZE 16

///////////////////////////////////
// tests for [[ hc_waves_per_eu ]]
///////////////////////////////////

[[ hc_waves_per_eu(1) ]]
[[ hc_grid_launch ]] void kernel_1_1(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_waves_per_eu(1, 4) ]]
[[ hc_grid_launch ]] void kernel_1_2(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_waves_per_eu(1, 4, "AMD:AMDGPU:8:0:3") ]]
[[ hc_grid_launch ]] void kernel_1_3(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

///////////////////////////////////
// tests for [[ hc_flat_workgroup_size ]]
///////////////////////////////////

[[ hc_flat_workgroup_size(64) ]]
[[ hc_grid_launch ]] void kernel_2_1(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_flat_workgroup_size(64, 1024) ]]
[[ hc_grid_launch ]] void kernel_2_2(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_waves_per_eu(1, 4, "AMD:AMDGPU:8:0:3") ]]
[[ hc_grid_launch ]] void kernel_2_3(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

///////////////////////////////////
// tests for [[ hc_max_workgroup_dim ]]
///////////////////////////////////

[[ hc_max_workgroup_dim(16) ]]
[[ hc_grid_launch ]] void kernel_3_1(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_max_workgroup_dim(16, 16) ]]
[[ hc_grid_launch ]] void kernel_3_2(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_max_workgroup_dim(16, 16, 16) ]]
[[ hc_grid_launch ]] void kernel_3_3(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}

[[ hc_max_workgroup_dim(16, 16, 16, "AMD:AMDGPU:8:0:3") ]]
[[ hc_grid_launch ]] void kernel_3_4(grid_launch_parm lp, int* x) {
  int idx = hc_get_workitem_absolute_id(0);
  x[idx] = idx;
}


// main program
int main() {

  const int sz = GRID_SIZE*TILE_SIZE;

  int* data1 = (int* )malloc(sz*sizeof(int));

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(sz*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(GRID_SIZE, 1);
  lp.group_dim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel_1_1(lp, data1_d);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, sz*sizeof(int));

  bool ret = true;

  for(int i = 0; i < sz; ++i) {
    ret &= (data1[i] == i);
  }

  hc::am_free(data1_d);
  free(data1);

  return !ret;

}
