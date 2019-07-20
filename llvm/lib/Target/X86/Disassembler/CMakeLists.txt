set(LLVM_LINK_COMPONENTS
  MCDisassembler
  Support
  X86Info
  )

add_llvm_library(LLVMX86Disassembler
  X86Disassembler.cpp
  X86DisassemblerDecoder.cpp
  )
