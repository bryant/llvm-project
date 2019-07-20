set(LLVM_LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  MC
  MCParser
  Support
  )

add_llvm_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
