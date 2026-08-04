import sys, os
import pytimeloop.timeloopfe.v4 as tl

problem = sys.argv[1]          # 'problem_qkt.yaml' or 'problem_v.yaml'
out_dir = sys.argv[2]
mesh_x = int(sys.argv[3])
mesh_y = int(sys.argv[4])
scratch_depth = int(sys.argv[5])   # words (16B/word at width=128,datawidth=8)
accum_depth = int(sys.argv[6])     # words (4B/word at width=32,datawidth=32) per PE instance

spec = tl.Specification.from_yaml_files("top.yaml.jinja2", jinja_parse_data={"problem": problem})

pe = spec.architecture.find("PE")
pe.spatial.meshX = mesh_x
pe.spatial.meshY = mesh_y

scratch = spec.architecture.find("Scratchpad")
scratch.attributes["depth"] = scratch_depth

accum = spec.architecture.find("Accumulator")
accum.attributes["depth"] = accum_depth

spec.mapper.algorithm = "random_pruned"
spec.mapper.num_threads = 4
spec.mapper.search_size = int(sys.argv[7]) if len(sys.argv) > 7 else 200000
spec.mapper.victory_condition = int(sys.argv[8]) if len(sys.argv) > 8 else 5000
spec.mapper.timeout = int(sys.argv[9]) if len(sys.argv) > 9 else 100000

os.makedirs(out_dir, exist_ok=True)
tl.call_mapper(spec, output_dir=out_dir, log_to=f"{out_dir}/log.txt")
stats = tl.parse_timeloop_output(spec, out_dir, "timeloop-mapper")

print("=== RESULT", problem, mesh_x, mesh_y, scratch_depth, accum_depth, "===")
print("cycles", stats.cycles)
print("utilization_pct", stats.percent_utilization)
print("energy_J", stats.energy)
print("area_m2", stats.area)
print("---mapping---")
print(stats.mapping)
