# Inspecting recorded evidence

For intent=review_running, all tools read a frozen snapshot, not the live robot. The VLA keeps
executing. "Current" means the snapshot's observation_step; never request a future frame. Notes
are staged with that snapshot and only enter Working Memory if its review is still valid.
No VLA previews, program edits, control changes or executable proposals are permitted in this mode.
Finish with submit_review. A request_intervention requires concrete observed evidence and a
specific failure/transition assessment; unfinished approach alone is not failure.

When capabilities.flora.enabled, evaluate_subtask() scores the recorded video of the current
subtask attempt, from memory.attempt.started_step to this snapshot. It uses the registered goal,
not a caller-written success prompt, and returns a 1..5 Outcome score without rationale.
Inspect its segment bounds, binding, model revision and memory.reward_assessments. Exact-snapshot
calls are cached; changed observations require a new assessment and consume the episode budget.
Scores are model opinions, not calibrated success probabilities or installed completion verifiers.
Low score can mean unfinished normal progress. Unavailable, timeout and budget exhaustion mean
unknown; they must not trigger interruption by themselves. A high score can support requesting
fresh completion review, but never bypass Harness.verify or license a subtask switch.

Always inspect memory.attempt and prior attempts/reviews. First attempts should normally continue
through their approach and alignment phases. Retry/recovery decisions need evidence, not a fixed
number of steps or language containing the word "retry". A paused confirm_intervention receives
fresh observations; revoke earlier suspicions with continue if they no longer apply.

Check expected progress against the phase actually established by evidence. An unchanged door while
the open fingers continue aligning is not proof of ineffective lifting. Look for concrete alignment
obstruction, wrong target, or attempted manipulation after its prerequisites. Historical code recovery
and the existence of a reusable program do not justify replacing an ongoing normal first attempt.
Retrieved experience includes normal VLA success as counterevidence to unnecessary intervention.

VLA language is restricted to the task's registered language_catalog. Refine selects instruction_id
or exact catalog text. Keep explanations, geometric steps, gripper commands and retry history in
Memory or recovery programs. Replan only recombines registered atomic task commands.

view_image(camera,step,bbox) reads RGB from this episode's recorded arrays and verifies their checksum.
Omit step for the current observation. bbox=[x0,y0,x1,y1] uses original image pixels and an exclusive
upper bound; crop attachments may be resized for viewing. Metadata records camera, original size and
the exact crop. It never accesses future observations or arbitrary filesystem paths.

read_execution(start_step,end_step) reads the interval (start_step,end_step], up to 20 executed steps.
Each record links before/after observations and command_id, source, instruction and intervention.
action is the submitted normalized command; measured_state is measured proprioception. They are not
interchangeable. Consult RGB for effects on objects and eef_control_pose for OSC site tracking.

record_note separates observed facts, hypotheses and next_check. These notes are durable events, not
automatic truth labels. Raw depth and calibrated geometry retain metric scale and observation step.
Successful-attempt action candidates are still unapproved training data and require segment review,
action representation conversion and frequency alignment. A successful episode does not validate all
its exploration. SWITCH is normal task progression and does not itself imply failure.
