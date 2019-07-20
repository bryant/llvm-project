set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  PowerPCDesc
  PowerPCInfo
  Support
  )

add_llvm_library(LLVMPowerPCAsmParser
  PPCAsmParser.cpp
  )
