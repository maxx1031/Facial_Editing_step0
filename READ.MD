python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0019_pair00_s0 \
  --instruction "Edit only the facial expression from happy to angry. Remove smile and cheek raise (deactivate AU12, AU6). Close mouth and remove teeth visibility (deactivate AU25, AU26). Lower and draw brows together (activate AU4). Tighten lower eyelids (activate AU7). Press and tighten lips horizontally (activate AU23, AU24). Slightly tense chin (mild AU17). Keep identity, age, gender, skin tone, hair, pose, lighting, and background unchanged. and lowered brows." \
  --note "v3 prompt test"


CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python /scratch3/f007yzf/repos/Step1X-Edit/inference.py --model_path /scratch3/f007yzf/models/step1x_v11 --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0000_pair00_s0/inputs --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0000_pair00_s0/outputs --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0000_pair00_s0/prompts.json --task_type edit
--version v1.1 \
--offload

python /scratch3/f007yzf/repos/Step1X-Edit/inference.py \
  --model_path /scratch3/f007yzf/models/step1x_v11 \
  --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_224938_p0019_pair00_s0/inputs \
  --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_224938_p0019_pair00_s0/outputs \
  --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_224938_p0019_pair00_s0/prompts.json \
  --task_type edit \
  --version v1.1 \
  --offload


CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /scratch3/f007yzf/repos/Step1X-Edit/inference.py --model_path /scratch3/f007yzf/models/step1x_v11 --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0101_pair00_s0/inputs --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0101_pair00_s0/outputs --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_230520_p0101_pair00_s0/prompts.json --task_type edit --version v1.1 --offload



python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0101_pair00_s0 \
  --instruction "Modify facial musculature from a Duchenne smile to an anger expression. Deactivate zygomatic major contraction (AU12) to remove lip corner elevation. Relax orbicularis oculi (AU6) to eliminate cheek raise and eye crinkling. Close the mouth by deactivating AU25 and remove upper lip elevation. Activate corrugator supercilii (AU4) to lower and medially draw the eyebrows. Increase eyelid tension via AU7. Activate orbicularis oris tightening (AU23/AU24) to compress the lips. Slightly activate mentalis (AU17) to raise the chin. Keep identity, age, gender, skin tone, lighting unchanged. Do not alter head pose, camera distance, hairstyle, or background. Only modify muscle activation patterns." \
  --note "v3 prompt test"


============== 101 Expression ===============
Change facial expression from a Duchenne smile to anger. Deactivate AU12 to remove lip-corner lift. Deactivate AU6 to remove cheek raise and eye crinkling. Deactivate AU25 to close the mouth and remove upper-lip elevation. Activate AU4 to lower and draw the eyebrows inward. Activate AU7 to increase lower-eyelid tension. Activate AU23 and AU24 to press and tighten the lips. Slightly activate AU17 to raise and tense the chin. Keep identity, age, gender, skin tone, lighting unchanged. Do not alter head pose, camera distance, hairstyle, or background. Only modify the facial Action Units above.

python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0101_pair00_s0 \
  --instruction "Change facial expression from a Duchenne smile to anger. Deactivate AU12 to remove lip-corner lift. Deactivate AU6 to remove cheek raise and eye crinkling. Deactivate AU25 to close the mouth and remove upper-lip elevation. Activate AU4 to lower and draw the eyebrows inward. Activate AU7 to increase lower-eyelid tension. Activate AU23 and AU24 to press and tighten the lips. Slightly activate AU17 to raise and tense the chin. Keep identity, age, gender, skin tone, lighting unchanged. Do not alter head pose, camera distance, hairstyle, or background. Only modify the facial Action Units above." \
  --note "v3 prompt test"



CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /scratch3/f007yzf/repos/Step1X-Edit/inference.py --model_path /scratch3/f007yzf/models/step1x_v11 --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_231921_p0101_pair00_s0/inputs --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_231921_p0101_pair00_s0/outputs --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_231921_p0101_pair00_s0/prompts.json --task_type edit --version v1.1 --offload


============== 48 Age ===============
Modify facial musculature to express surprise. Activate frontalis muscle to raise inner and outer brows (AU1 + AU2), creating horizontal forehead wrinkles. Activate levator palpebrae superioris to widen the eyes (AU5). Relax orbicularis oculi to prevent eyelid drooping. Release lip pressor tension (deactivate AU24).Relax depressor anguli oris (deactivate AU15). Activate jaw depressor muscles to drop the mandible (AU26).Part the lips naturally (AU25) without stretching corners laterally. Keep identity, age, gender, skin tone, lighting unchanged. Do not modify hairstyle, head pose, facial structure, texture, or image style. Only alter muscle activations corresponding to the specified Action Units.

python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0048_pair00_s0 \
  --instruction "Modify facial musculature to express surprise. Activate frontalis muscle to raise inner and outer brows (AU1 + AU2), creating horizontal forehead wrinkles. Activate levator palpebrae superioris to widen the eyes (AU5). Relax orbicularis oculi to prevent eyelid drooping. Release lip pressor tension (deactivate AU24).Relax depressor anguli oris (deactivate AU15). Activate jaw depressor muscles to drop the mandible (AU26).Part the lips naturally (AU25) without stretching corners laterally. Keep identity, age, gender, skin tone, lighting unchanged. Do not modify hairstyle, head pose, facial structure, texture, or image style. Only alter muscle activations corresponding to the specified Action Units." \
  --note "v3 prompt test"


  ============== 51 Sex ===============
python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0051_pair00_s0 \
  --instruction "Relax AU4 to remove brow lowering and medial brow contraction, deactivate AU15 and AU17 to release lower-lip tension and remove lip compression, activate AU12 to lift lip corners upward and laterally, activate AU6 to raise cheeks and narrow the eye aperture slightly, activate mild AU25 and AU26 for gentle mouth opening with upper-teeth visibility, keep activation symmetric on both sides, and keep identity, age, gender, skin tone, and lighting unchanged while only changing expression-related Action Units."

Relax AU4 to remove brow lowering and medial brow contraction, deactivate AU15 and AU17 to release lower-lip tension and remove lip compression, activate AU12 to lift lip corners upward and laterally, activate AU6 to raise cheeks and narrow the eye aperture slightly, activate mild AU25 and AU26 for gentle mouth opening with upper-teeth visibility, keep activation symmetric on both sides, and keep identity, age, gender, skin tone, and lighting unchanged while only changing expression-related Action Units.


Modify facial musculature to express a Duchenne smile. Deactivate corrugator supercilii and depressor supercilii to relax brow lowering (set AU4 to 0), smoothing the glabellar region. Deactivate depressor anguli oris (AU15), mentalis (AU17), and lip pressor (AU24) to release lower-face tension and remove lip compression. Activate zygomaticus major (AU12) to strongly lift lip corners upward and laterally. Activate orbicularis oculi pars orbitalis (AU6) to raise the cheeks and produce infraorbital compression, slightly narrowing the eye aperture.
Activate mild AU25 (lips part) and AU26 (jaw drop) to create natural mouth opening with visible upper teeth.
Maintain symmetric activation across both sides of the face. Keep identity, age, gender, skin tone, lighting unchanged. Do not modify hairstyle, head pose, facial structure, texture, or image style.Only alter muscle activations corresponding to the specified Action Units.

python /scratch3/f007yzf/repos/Step1X-Edit/inference.py --model_path /scratch3/f007yzf/models/step1x_v11 --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_233148_p0051_pair00_s0/inputs --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_233148_p0051_pair00_s0/outputs --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_233148_p0051_pair00_s0/prompts.json --task_type edit --version v1.1 --offload

  ============== 105 Sex ===============
python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id p0105_pair00_s0 \
  --instruction "Modify only facial muscle activations to transform the expression from a Duchenne smile to a frown.Deactivate zygomaticus major (remove upward lip corner pull; eliminate AU12). Relax orbicularis oculi (remove cheek raising and eye crinkling; eliminate AU6). Close the mouth by relaxing jaw depressor muscles (remove AU25 and AU26).Activate corrugator supercilii and depressor supercilii to draw the eyebrows downward and medially, forming vertical glabellar lines (activate AU4).Activate orbicularis oculi (palpebral portion) to tighten lower eyelids slightly (activate AU7). Activate depressor anguli oris to pull lip corners downward (activate AU15). Engage orbicularis oris to compress the lips (activate AU23). Slightly raise and tense the chin via mentalis activation (activate AU17).Keep identity, age, gender, skin tone, lighting unchanged. Only modify expression-related muscle activations."

  ============== 105 Sex ===============
  python /scratch3/f007yzf/flux_face_emotion/prepare_step1x_case.py \
  --image-id f0000_pair00_s0 \
  --instruction "Relax the corrugator supercilii to remove brow lowering (deactivate AU4). Smooth the glabellar region and eliminate vertical forehead tension. Release orbicularis oris compression to remove lip pressing (deactivate AU24). Relax mentalis to remove chin tension (deactivate AU17). Soften lower eyelid tension (reduce AU7). Bilaterally contract zygomaticus major to pull lip corners upward (activate AU12). Engage orbicularis oculi (pars orbitalis) to raise cheeks and create mild lower-lid compression (activate AU6). Part the lips slightly (activate AU25). Introduce slight mandibular depression for visible upper teeth (activate AU26). Allow mild upper lip elevation from smile mechanics (mild AU10)."

python /scratch3/f007yzf/repos/Step1X-Edit/inference.py --model_path /scratch3/f007yzf/models/step1x_v11 --input_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_235001_f0000_pair00_s0/inputs --output_dir /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_235001_f0000_pair00_s0/outputs --json_path /scratch3/f007yzf/flux_face_emotion/mvp_step1x_case_runs/20260215_235001_f0000_pair00_s0/prompts.json --task_type edit --version v1.1 --offload