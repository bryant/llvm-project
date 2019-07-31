add_llvm_library(LLVMAMDGPUUtils
  AMDGPUBaseInfo.cpp
  AMDKernelCodeTUtils.cpp
  AMDGPUAsmUtils.cpp
  AMDGPUPALMetadata.cpp

  LINK_COMPONENTS

  BinaryFormat
  Core
  MC
  Support
  )
