import torch

# 1. Your specific file paths
old_checkpoint_path = '../experiments/ablations/baseline_ce/baseline_ce_09-05-2026/model_acc_0.7590_ep_70.pth'
new_checkpoint_path = '../experiments/ablations/baseline_ce/baseline_ce_09-05-2026/model_acc_0.7590_ep_70_new.pth'

print(f"Loading old checkpoint: {old_checkpoint_path} ...")
checkpoint = torch.load(old_checkpoint_path, map_location='cpu', weights_only=False)

# Handle both raw state_dicts and nested checkpoint dictionaries
is_nested = 'model_g_sd' in checkpoint
state_dict = checkpoint['model_g_sd'] if is_nested else checkpoint

new_state_dict = {}

print("Translating keys to the new architecture...")
for key, value in state_dict.items():
    new_key = key
    
    # -------------------------------------------------------------
    # Translation Map for SpatialFeatureExtractor
    # -------------------------------------------------------------
    if 'student_extractor.stem.' in key:
        new_key = new_key.replace('student_extractor.stem.0', 'student_extractor.stem_in.0')
        new_key = new_key.replace('student_extractor.stem.1', 'student_extractor.stem_in.1')
        new_key = new_key.replace('student_extractor.stem.3', 'student_extractor.stem_out.0')
        new_key = new_key.replace('student_extractor.stem.4', 'student_extractor.stem_out.1')
        new_key = new_key.replace('student_extractor.stem.5', 'student_extractor.stem_out.2')
        
    # -------------------------------------------------------------
    # Translation Map for CustomOCR
    # -------------------------------------------------------------
    elif 'student_ocr.stem.' in key:
        new_key = new_key.replace('student_ocr.stem.0', 'student_ocr.stem_down.0')
        new_key = new_key.replace('student_ocr.stem.1', 'student_ocr.stem_down.1')
        new_key = new_key.replace('student_ocr.stem.3', 'student_ocr.stem_align.0')
        new_key = new_key.replace('student_ocr.stem.4', 'student_ocr.stem_align.1')
        new_key = new_key.replace('student_ocr.stem.5', 'student_ocr.stem_align.2')
        new_key = new_key.replace('student_ocr.stem.6', 'student_ocr.stem_align.3')
        new_key = new_key.replace('student_ocr.stem.7', 'student_ocr.stem_align.4')
        new_key = new_key.replace('student_ocr.stem.8', 'student_ocr.stem_align.5')

    new_state_dict[new_key] = value

# Repackage the checkpoint preserving all other data (optimizer, epoch, etc.)
if is_nested:
    checkpoint['model_g_sd'] = new_state_dict
else:
    checkpoint = new_state_dict

print(f"Saving translated weights to {new_checkpoint_path}...")
torch.save(checkpoint, new_checkpoint_path)
print("Done! You can now load this translated file into your new code.")