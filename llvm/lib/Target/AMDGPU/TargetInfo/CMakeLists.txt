set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMAMDGPUInfo
  AMDGPUTargetInfo.cpp
  )
