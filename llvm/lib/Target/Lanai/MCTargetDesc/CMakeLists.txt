set(LLVM_LINK_COMPONENTS
  LanaiInfo
  MC
  MCDisassembler
  Support
  )

add_llvm_library(LLVMLanaiDesc
  LanaiAsmBackend.cpp
  LanaiELFObjectWriter.cpp
  LanaiInstPrinter.cpp
  LanaiMCAsmInfo.cpp
  LanaiMCCodeEmitter.cpp
  LanaiMCExpr.cpp
  LanaiMCTargetDesc.cpp
)
