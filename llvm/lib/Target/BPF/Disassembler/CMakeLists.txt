set(LLVM_LINK_COMPONENTS
  BPFInfo
  MCDisassembler
  Support
  )

add_llvm_library(LLVMBPFDisassembler
  BPFDisassembler.cpp
)

