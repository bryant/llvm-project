set(LLVM_LINK_COMPONENTS
  BPFInfo
  MC
  Support
  )

add_llvm_library(LLVMBPFDesc
  BPFMCTargetDesc.cpp
  BPFAsmBackend.cpp
  BPFInstPrinter.cpp
  BPFMCCodeEmitter.cpp
  BPFELFObjectWriter.cpp
  )
