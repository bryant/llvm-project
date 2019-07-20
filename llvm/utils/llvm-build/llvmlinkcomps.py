from llvmbuild.main import LLVMProjectInfo
from os import pardir, path

root = path.abspath(path.join(path.dirname(__file__), pardir, pardir))
proj = LLVMProjectInfo.load_from_path(root, root)

for c in proj.component_infos:
    if c.type_name == "TargetGroup" or not hasattr(c, "required_libraries"):
        continue
    cmakelists = path.join(root, c.subpath.lstrip("/"), "CMakeLists.txt")
    cm = open(cmakelists).read()  # Never fails.

    if "LINK_COMPONENTS" in cm:  # Defer to existing specified deps.
        continue

    print("Setting link comps for ", c.name, c.subpath)

    req, magic = [], []
    for r in c.required_libraries:
        if r.lower() == "all-targets":
            magic += ["AllTargetsAsmParsers", "AllTargetsCodeGens",
                    "AllTargetsDescs", "AllTargetsInfos"]
        elif r.lower() in "native nativecodegen engine".split():
            magic.append(r)
        else:
            req.append(r)

    join = lambda ss: "".join("\n  " + s for s in sorted(ss))
    linkcomps = "set(LLVM_LINK_COMPONENTS%s%s\n  )\n" % (join(req), join(magic))
    open(cmakelists, "w").write(linkcomps + "\n" + cm)

    print("Wrote %r" % cmakelists)
