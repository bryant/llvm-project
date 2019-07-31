from __future__ import print_function
from llvmbuild.main import LLVMProjectInfo
from os import pardir, path
import re

root = path.abspath(path.join(path.dirname(__file__), pardir, pardir))
proj = LLVMProjectInfo.load_from_path(root, root)
proj.validate_components()  # Populates component_info_map.

for c in proj.component_infos:
    if c.type_name == "TargetGroup" or not hasattr(c, "required_libraries"):
        continue
    cmakelists = path.join(root, c.subpath.lstrip("/"), "CMakeLists.txt")
    cm = open(cmakelists).read()  # Never fails.

    print("Setting link comps for", c.name, c.subpath, repr(cmakelists))

    if "LINK_COMPONENTS" in cm:  # Defer to existing specified deps.
        print("*LINK_COMPONENTS already present.")
        continue

    req, magic = [], []
    for r in c.required_libraries:
        if r.lower() == "all-targets":
            magic += ["AllTargetsAsmParsers", "AllTargetsCodeGens",
                    "AllTargetsDescs", "AllTargetsInfos"]
        elif r.lower() in "native nativecodegen engine".split():
            magic.append(r)
        else:
            req.append(proj.component_info_map[r].get_library_name())

    pat = r"add_llvm_(?:tool|library|executable|target)\(.+?\)"
    flags = re.M | re.DOTALL
    res = re.search(pat, cm, flags=flags)
    if res is None:
        print("No add_llvm found in", c.get_library_name(), c.subpath)

    def repl(m):
        indent = lambda s: " " * 2 + s
        try:
            head, tail = m.group(0).rsplit("\n", 1)
        except ValueError:
            head, tail = m.group(0)[:-1], indent(")")
        assert tail.endswith(")")
        join = lambda ss: "".join("\n" + indent(s) for s in sorted(ss))
        linkcomps = "\n\n" + indent("LINK_COMPONENTS")

        if len(req) + len(magic) == 0:
            linkcomps += "\n\n" + indent("# No component dependencies.\n")
        else:
            linkcomps += "\n%s%s\n" % (join(req), join(magic))
        return head + linkcomps + tail
    cm = re.sub(pat, repl, cm, count=1, flags=flags)
    open(cmakelists, "w").write(cm)

    print("Wrote LINK_COMPONENTS.")
