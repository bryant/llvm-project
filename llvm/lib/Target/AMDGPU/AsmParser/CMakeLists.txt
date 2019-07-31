add_llvm_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp

  LINK_COMPONENTS

  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  MC
  MCParser
  Support
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
