//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME this file will place C++AMP Runtime implementation (HSA version)
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <amp_runtime.h>

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {}

namespace Concurrency {

class CPUManager : public AMPManager
{
    std::shared_ptr<AMPAllocator> aloc;
    std::shared_ptr<AMPAllocator> init();
public:
    CPUManager() : AMPManager(L"fallback") {
        des = L"CPU Fallback";
        mem = 0;
        is_double_ = true;
        is_limited_double_ = true;
        cpu_shared_memory = true;
        emulated = false;
    }
    void* create(size_t count, void *data, bool hasSrc) override {
        if (!hasSrc)
            return data;
        else
            return aligned_alloc(0x1000, count);
    }
    void release(void *data) override {
        ::operator delete(data);
    }
    std::shared_ptr<AMPAllocator> createAloc() override {
        if (!aloc)
            aloc = init();
        return aloc;
    }
};

class CPUAllocator : public AMPAllocator
{
    std::map<void*, void*> addrs;
public:
    CPUAllocator(std::shared_ptr<AMPManager> pMan) : AMPAllocator(pMan) {}
  void amp_write(void *data) override {
      obj_info obj = Man->device_data(data);
      if (obj.device != data)
          memmove(obj.device, data, obj.count);
  }
  void amp_read(void *data) override {
      obj_info obj = Man->device_data(data);
      if (obj.device != data)
          memmove(data, obj.device, obj.count);
  }
  void amp_copy(void *dst, void *src, size_t n) override {
      obj_info obj = Man->device_data(src);
      if (obj.device != dst)
          memmove(dst, src, obj.count);
  }
  void PushArg(void* kernel, int idx, rw_info& data) override {
      obj_info obj = Man->device_data(data.data);
      if (data.data == obj.device)
          return;
      auto it = addrs.find(data.data);
      bool find = it != std::end(addrs);
      if (!kernel && !find) {
          addrs[obj.device] = data.data;
          data.data = obj.device;
      } else if (kernel && find) {
          data.data = it->second;
          addrs.erase(it);
      }
  }
};

std::shared_ptr<AMPAllocator> CPUManager::init() {
    return std::shared_ptr<AMPAllocator>(new CPUAllocator(shared_from_this()));
}

class CPUContext : public AMPContext
{
public:
    CPUContext() {
        auto Man = std::shared_ptr<AMPManager>(new CPUManager);
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
    }
};


static CPUContext ctx;

} // namespace Concurrency


///
/// kernel compilation / kernel launching
///

extern "C" void *GetContextImpl() {
  return &Concurrency::ctx;
}
