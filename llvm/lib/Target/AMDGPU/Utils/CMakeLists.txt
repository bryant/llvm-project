set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Core
  MC
  Support
  )

add_llvm_library(LLVMAMDGPUUtils
  AMDGPUBaseInfo.cpp
  AMDKernelCodeTUtils.cpp
  AMDGPUAsmUtils.cpp
  AMDGPUPALMetadata.cpp
  )
