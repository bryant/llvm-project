set(LLVM_LINK_COMPONENTS
  MCDisassembler
  PowerPCInfo
  Support
  )

add_llvm_library(LLVMPowerPCDisassembler
  PPCDisassembler.cpp
  )
