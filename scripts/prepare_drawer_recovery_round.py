"""Build immutable targeted code revisions and replay specs from the atomic pilot."""

import argparse
import json
import shutil
from pathlib import Path

from vla_agent.skill_packages import build_package, check_package, read_files
from vla_agent.skills import SkillLibrary


def main(root):
    repo = Path(__file__).resolve().parents[1]
    previous = repo / "runs/opendrawer-atomic10-20261104"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(previous / "retained/OpenDrawer/skills", root / "skills")
    (root / "specs").mkdir()
    (root / "episodes").mkdir()
    ids = {
        4: "open-drawer-from-misaligned-handle",
        5: "drawer-pull-from-handle-hold",
        6: "open-drawer-from-misaligned-open-gripper",
        7: "open-drawer-from-missed-grasp",
    }
    changes = {
        4: "Retreat upward 6 cm before wrist reorientation; retain pose tolerances.",
        5: "Extend bounded pull travel from 0.30 to 0.45 m; preserve slip, action and time bounds.",
        6: "Use >=4 pixel handle strips, skip individually invalid fragments; preserve geometric gates.",
        7: "Project expected 3D points into current calibrated views, jointly match both endpoints; preserve bar checks.",
        8: "Execute the previously submitted but unexecuted v002 candidate unchanged, without a new model call.",
    }
    specs = []
    for seed in (4, 5, 6, 7, 8):
        library = SkillLibrary(root / "skills", "accumulate")
        ep = previous / "episodes" / f"OpenDrawer-seed{seed:02d}"
        rows = [json.loads(line) for line in (ep / "events.jsonl").open()]
        submissions = [
            e["arguments"]
            for e in rows
            if e["kind"] == "codex_tool"
            and e["name"] == "submit_intervention"
            and e["arguments"]["kind"] == "code"
        ]
        if seed == 8:
            directory = next(ep.glob("workspaces/*/versions/" + submissions[0]["program_version"]))
            manifest = json.loads((directory / "manifest.json").read_text())
            package = manifest["package"]
            files = read_files(directory, package)
            parent = dict(
                id=manifest["skill_name"],
                preconditions=[],
                required_capabilities=["native_action12"],
                parameter_schema=manifest["parameter_schema"],
                parameter_sources=manifest["parameter_sources"],
                known_failures=[],
            )
        else:
            parent = next(s for s in library.index if s["id"] == ids[seed])
            package = parent["package"]
            files = dict(parent["package_files"])
            module = package["module"]
            path = f"code/{module}/"
            if seed == 4:
                files[path + "controller.py"] = files[path + "controller.py"].replace(
                    'retreat=o["eef"]+RETREAT_M*n', 'retreat=o["eef"]+RETREAT_M*n+np.array([0.,0.,0.06])'
                )
            elif seed == 5:
                files[path + "params.py"] = files[path + "params.py"].replace(
                    "MAX_PULL_M = 0.30", "MAX_PULL_M = 0.45"
                )
            elif seed == 6:
                code = files[path + "perception.py"]
                code = code.replace("len(run)<3", "len(run)<4").replace("len(run)-2,3", "len(run)-3,4")
                code = code.replace(
                    "p,d=geom(ctx,camera,[xx,y,xx+3,y+1])",
                    "try: p,d=geom(ctx,camera,[xx,y,xx+4,y+1])\n                except ValueError: continue",
                )
                files[path + "perception.py"] = code
            elif seed == 7:
                code = files[path + "perception.py"]
                start = code.index("def track(")
                end = code.index("\ndef pads(", start)
                code = code[:start] + (root.parent / "projected_tracker.py").read_text() + code[end:]
                files[path + "perception.py"] = code
            files["SKILL.md"] += (
                "\nDevelopment revision: " + changes[seed] + " Not yet physically validated.\n"
            )
            package = build_package(package["name"], files, package["uses"])
        assert check_package(package, files)["ok"]
        item = library.record_candidate(
            files["code/run.py"],
            "PandaOmron",
            "task_success",
            dict(source="offline_failure_analysis", change=changes[seed], verified=False),
            "target",
            parent_skill=parent,
            description="OpenDrawer | " + changes[seed],
            skill_name=package["name"],
            package=package,
            package_files=files,
        )
        version = next((e for e in rows if e["kind"] == "code_version"), None)
        prefix = version["parameters"].get("observation_step") if version else 710
        if prefix is None:
            prefix = next(
                e["binding"]["observation_step"]
                for e in rows
                if e["kind"] == "intervention_plan" and e["plan"]["kind"] == "code"
            )
        spec = dict(
            seed=seed,
            source=str(ep),
            config=str(previous / "episodes" / f"OpenDrawer-seed{seed:02d}.yaml"),
            prefix_steps=prefix,
            library=str(root / "skills"),
            skill_id=item["id"],
            skill_version=item["version"],
            parameters=submissions[0]["parameters"],
            change=changes[seed],
        )
        (root / "specs" / f"seed{seed:02d}.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        )
        specs.append(spec)
    (root / "plan.json").write_text(
        json.dumps(
            dict(kind="exact-prefix-recovery-development", cases=specs, fresh_agent_sr=False, model_calls=0),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(specs, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
