set(LLVM_LINK_COMPONENTS
  AVRInfo
  MCDisassembler
  Support
  )

add_llvm_library(LLVMAVRDisassembler
  AVRDisassembler.cpp
)

