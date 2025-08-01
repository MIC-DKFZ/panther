# FOR MultiTalent V1 V2 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"

bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"

bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"

bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
#

## FOR Task1 MultiTalentV1 1e3 training 
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalent_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalent_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalent_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalent_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalent_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
## FOR Task1 MultiTalentV2 1e3 training 
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
## FOR Task2 MultiTalentV2 1e3 training 
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"


# FOR PancCTPretrain
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"

bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTPretrain -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
#

# FOR Task1 tumorOnly onehot training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 904 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 904 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 904 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 904 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 904 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
# FOR Task2 tumorOnly onehot training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
# FOR Task2 tumorOnly onehot 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 905 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"



# FOR Task1 softmax training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 907 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 907 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 907 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 907 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 907 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
# FOR Task2 softmax training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 908 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 908 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 908 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 908 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 908 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"


# FOR Panther_Combined 903 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlans"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlans"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlans"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlans"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlans"
#

# FOR Panther Task1 PancCTPretrain 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3  -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth" 
#
# FOR Panther Task2 PancCTPretrain 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTPretrain_1e3  -tr nnUNetTrainer1e3  -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth" 
#
# FOR Panther Task1 PancCTPretrainMultiTalentV2 Cascade 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3  -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth" 
# bs4
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1_bs4 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3  -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth" 

# FOR Panther Task1 PancCTPretrainMultiTalentV2 Cascade 1e3 training FOLD ALL
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 all -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_all  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
#####
# FOR Panther Task2 PancCTPretrainMultiTalentV2 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 902 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3  -tr nnUNetTrainer1e3  -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth" 
#

# FOR Panther_Combined 903 MultiTalentV1 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
#
# FOR Panther_Combined 903 MultiTalentV2 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
#
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 --c"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 --c"
#
# FOR Panther_Combined 906 MultiTalentV2 1e3 onlyTm training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
# FOR Panther_Combined 906 MultiTalentV2 1e3 onlyTm Task1inAll training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
## FOR Panther_Combined 906 MultiTalentV2 1e3 onlyTm training FOLD ALL only Task2 VAL
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 all -p nnUNetResEncUNetLPlansMultiTalentV2_all -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"
#
# FOR Panther_Combined 903 PancCTMultiTalentV2 Cascade 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
# FOR Panther_Combined 903 PancCTMultiTalentV2 Cascade 1e3 Task1inAllFolds training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
# FOR Panther_Combined 903 PancCTMultiTalentV2 Cascade 1e3 training FOLD ALL
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 903 3d_fullres_iso1x1x1 all -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_all -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"

bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=10G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 --val --val_best"

######
# FOR Panther_Combined 906 PanCTMultiTalentV2 Cascade 1e3 onlyTm training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 906 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset999_PancreasPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_final.pth"
#
git remote add origin https://github.com/ofdurugol/Panther_dkfz.git
git branch -M main
git push -u origin main
# FOR Panther Task1_50Annotated 911 MultiTalentV1 1e3 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 911 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlansMultiTalentV1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 911 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlansMultiTalentV1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 911 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlansMultiTalentV1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 911 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 911 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlansMultiTalentV1_1e3 -tr nnUNetTrainer1e3 -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth"
#

# Cascade PancCTPretrain on MultiTalentV2
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 999 3d_fullres_bs4 all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/nnUNetV2/MT_results/nnUNet_trained_models/Dataset903_MT_big_withholdout/MultiTalent_trainer_multistems_6000ep__nnUNetResEncUNetLPlansIso1x1x1_bs16__3d_fullres/fold_all/checkpoint_final.pth"

# Complete missing validation folder in fold 3 of 901 training
bsub -R "tensorcore3d" -gpu num=1:j_exclusive=yes:gmem=33G -q gpu-debian ". ~/panther.sh && nnUNetv2_train 901 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlansMultiTalentV2 --val"



# GT Spacing Check PANTHER2 
python check_spacing.py --f /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations/10303.nii.gz
# VAL Spacing Check PANTHER2 
python check_spacing.py --f /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations/10303.nii.gz

python check_spacing.py --f /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations/10002_0001.nii.gz
python check_spacing.py --f /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansPancCTPretrain__3d_fullres_iso1x1x1/fold_0/validation/10002_0001.nii.gz

# Evaluate PANTHER Metrics for Task1 Task2 Scratch Scratch_iso1x1x1
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/y033f/benchmarkability_trained_models/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_Scratch.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/y033f/benchmarkability_trained_models/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_Scratch_iso1x1x1.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/y033f/benchmarkability_trained_models/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_Scratch.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/y033f/benchmarkability_trained_models/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_Scratch_iso1x1x1.json --verbose

# Evaluate PANTHER Metrics for Task1 MultiTalents
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_MultiTalentV1.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres/ --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_MultiTalentV2.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_MultiTalentV1_iso1x1x1.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_MultiTalentV2_iso1x1x1.json --verbose

# Evaluate PANTHER Metrics for Task2 MultiTalents
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_MultiTalentV1.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_MultiTalentV2.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_MultiTalentV1_iso1x1x1.json --verbose

python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_MultiTalentV2_iso1x1x1.json --verbose

# Evaluate PANTHER Metrics for Task1 PancreasPretrain
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansPancCTPretrain__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_PancreasPretrain.json --verbose
# Evaluate PANTHER Metrics for Task1 PancreasPretrain 1e3
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTPretrain_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_PancreasPretrain_1e3.json --verbose
# Evaluate PANTHER Metrics for Task1 PancCTMultiTalentV2 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_PancCTMultiTalentV2_iso1x1x1x_1e3.json --verbose

# Evaluate PANTHER Metrics for Task2 PancreasPretrain
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer__nnUNetResEncUNetLPlansPancCTPretrain__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_PancreasPretrain.json --verbose
# Evaluate PANTHER Metrics for Task2 PancreasPretrain 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTPretrain_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_PancreasPretrain_1e3.json --verbose
# Evaluate PANTHER Metrics for Task2 PancCTMultiTalentV2 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_902_PancCTMultiTalentV2_iso1x1x1_1e3.json --verbose

# Evaluate PANTHER Metrics for Task1&2_COMBINED separately
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --include "_0001.nii.gz" --save_path panthermetrics_903_Combined_Task1_iso1x1x1.json --verbose
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --save_path panthermetrics_903_Combined_Task2_iso1x1x1.json --verbose

# Evaluate nnunet Metrics for Combined Task1
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task1_iso1x1x1.json --verbose
# Evaluate nnunet Metrics for Combined Task2
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task2_iso1x1x1.json --verbose
# Evaluate nnunet Metrics for Combined Task1 MultiTalentV1 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task1_MultiTalentV1_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 MultiTalentV1 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV1_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task2_MultiTalentV1_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task1 MultiTalentV2 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task1_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 MultiTalentV2 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task2_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task1 MultiTalentV2 1e3 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset906_PANTHER_COMBINED_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_906_Combined_Task1_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 MultiTalentV2 1e3 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset906_PANTHER_COMBINED_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_906_Combined_Task2_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task1 PancCTMultiTalentV2 Cascade 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task1_PancCTMultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 PancCTMultiTalentV2 Cascade 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_iso1x1x1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task2_PancCTMultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 PancCTMultiTalentV2 Cascade Task1inAll 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset903_PANTHER_COMBINED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_Task1inAll__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset903_PANTHER_COMBINED/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_903_Combined_Task2_PancCTMultiTalentV2_Task1inAll_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task1 PancCTMultiTalentV2 Cascade 1e3 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset906_PANTHER_COMBINED_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3.json__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations --include "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_906_Combined_Task1_PancCTMultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 PancCTMultiTalentV2 Cascade 1e3 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset906_PANTHER_COMBINED_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2_1e3.json__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_906_Combined_Task2_PancCTMultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Combined Task2 PancCTMultiTalentV2 Cascade Task1inAll 1e3 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset906_PANTHER_COMBINED_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_Task1inAll__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations --exclude "_0001.nii.gz" --detailed_eval --save_path nnunetmetrics_906_Combined_Task2_MultiTalentV2_Task1inAll_iso1x1x1_1e3.json --verbose

# Evaluate nnunet Metrics for Task1 PancreasPretrain
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansPancCTPretrain__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --detailed_eval --save_path nnunetmetrics_901_PancreasPretrain.json --verbose

# Evaluate PANTHER Metrics for Task1_50Annotated MultiTalentV1 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset911_PANTHER_N_UNLABELED/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV1_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset911_PANTHER_N_UNLABELED/gt_segmentations --save_path panthermetrics_911_MultiTalentV1_iso1x1x1_1e3.json --verbose

# Evaluate PANTHER Metrics for Task1 softmax
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset907_PANTHER_softmax/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_907_softmax_iso1x1x1.json --verbose
# Evaluate PANTHER Metrics for Task1 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset904_PANTHER_onlyTumor/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset904_PANTHER_onlyTumor/gt_segmentations --save_path panthermetrics_904_onlyTumor_iso1x1x1.json --verbose
# Evaluate nnunet Metrics for Task1 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset904_PANTHER_onlyTumor/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset904_PANTHER_onlyTumor/gt_segmentations --detailed_eval --save_path panthermetrics_904_onlyTumor_iso1x1x1.json --verbose

# Evaluate PANTHER Metrics for Task2 softmax
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset908_PANTHER_HR_softmax/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --save_path panthermetrics_908_softmax_iso1x1x1.json --verbose
# Evaluate PANTHER Metrics for Task2 onlyTm
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset905_PANTHER_HR_onlyTumor/nnUNetTrainer__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset905_PANTHER_HR_onlyTumor/gt_segmentations --save_path panthermetrics_905_onlyTumor_iso1x1x1.json --verbose
# Evaluate PANTHER Metrics for Task2 onlyTm 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset905_PANTHER_HR_onlyTumor/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_preprocessed/Dataset905_PANTHER_HR_onlyTumor/gt_segmentations --save_path panthermetrics_905_onlyTumor_iso1x1x1_1e3.json --verbose

# Evaluate PANTHER Metrics for Task1 MultiTalentV2 iso1x1x1 1e3
python evaluate_local.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --save_path panthermetrics_901_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Task1 MultiTalentV2 iso1x1x1 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset901_PANTHER/gt_segmentations --detailed_eval --save_path nnunetmetrics_901_MultiTalentV2_iso1x1x1_1e3.json --verbose
# Evaluate nnunet Metrics for Task2 MultiTalentV2 iso1x1x1 1e3
python evaluate_local_v2.py --pred_dir /dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalentV2_1e3__3d_fullres_iso1x1x1 --gt_dir /dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed/Dataset902_PANTHER_HR/gt_segmentations --detailed_eval --save_path nnunetmetrics_902_MultiTalentV2_iso1x1x1_1e3.json --verbose

nnUNetv2_predict_from_modelfolder -i /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/imagesTs -o /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/predictionsTs_PancCTPretrain_iso1x1x1_1e3 -m /home/o644l/remote_files/nnunet_results_folder/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_iso1x1x1


/dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results/Dataset901_PANTHER/nnUNetTrainer__nnUNetResEncUNetLPlansPancCTPretrain__3d_fullres_iso1x1x1


python qc_nii.py /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/imagesTs/ /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/predictionsTs_PancCTPretrain_iso1x1x1_1e3/ 

python qc_nii.py /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/imagesTs/ /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/predictionsTs_PancCTPretrain_iso1x1x1_1e3/ qc_results.csv /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/imagesTr /home/o644l/projects/panther/nnUNet_raw/Dataset901_PANTHER/labelsTr




export CUDA=11.6
export PATH=/usr/local/lib:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export CPATH=/usr/local/lib:$CPATH
export PATH=/usr/local/cuda-${CUDA}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA}
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1

. /home/o644l/miniforge3/etc/profile.d/conda.sh
conda activate panther

export nnUNet_raw="/dkfz/cluster/gpu/data/OE0441/o644l/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/dkfz/cluster/gpu/data/OE0441/y033f/benchmarkability_data/nnUNet_preprocessed"
export nnUNet_results="/dkfz/cluster/gpu/checkpoints/OE0441/o644l/nnUNet_results"
export DATASET_LOCATION="/dkfz/cluster/gpu/data/OE0441/o644l"
export EXPERIMENT_LOCATION="/dkfz/cluster/gpu/checkpoints/OE0441/o644l"


pretrained:
{
    "configurations": {
        "3d_fullres": {
            "architecture": {
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ],
                "arch_kwargs": {
                    "conv_bias": true,
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "dropout_op": null,
                    "dropout_op_kwargs": null,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ],
                    "kernel_sizes": [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "n_blocks_per_stage": [
                        1,
                        3,
                        4,
                        6,
                        6,
                        6
                    ],
                    "n_conv_per_stage_decoder": [
                        1,
                        1,
                        1,
                        1,
                        1
                    ],
                    "n_stages": 6,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": true
                    },
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "affine": true,
                        "eps": 1e-05
                    },
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ]
                    ]
                },
                "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
            },
            "batch_dice": true,
            "batch_size": 24,
            "data_identifier": "znorm_3d_fullres",
            "median_image_size_in_voxels": [
                450.0,
                398.5,
                400.0
            ],
            "normalization_schemes": [
                "ZScoreNormalization"
            ],
            "patch_size": [
                192,
                192,
                192
            ],
            "preprocessor_name": "DefaultPreprocessor",
            "resampling_fn_data": "resample_torch_fornnunet",
            "resampling_fn_data_kwargs": {
                "force_separate_z": false,
                "is_seg": false,
                "memefficient_seg_resampling": false,
                "num_threads": 8
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "force_separate_z": null,
                "is_seg": false,
                "order": 1,
                "order_z": 0
            },
            "resampling_fn_seg": "resample_torch_fornnunet",
            "resampling_fn_seg_kwargs": {
                "force_separate_z": false,
                "is_seg": true,
                "memefficient_seg_resampling": false,
                "num_threads": 8
            },
            "spacing": [
                1.0,
                1.0,
                1.0
            ],
            "use_mask_for_norm": [
                false
            ]
        }
    },
    "dataset_name": "nativemultistem",
    "experiment_planner_used": "1x1x1",
    "foreground_intensity_properties_per_channel": {
        "0": {
            "max": 6739.0,
            "mean": 63.436882,
            "median": 96.0,
            "min": -1570.9453,
            "percentile_00_5": -937.0,
            "percentile_99_5": 275.0,
            "std": 175.48
        }
    },
    "image_reader_writer": "SimpleITKIO",
    "label_manager": "LabelManager",
    "original_median_shape_after_transp": [
        103,
        512,
        512
    ],
    "original_median_spacing_after_transp": [
        5.0,
        0.712890625,
        0.712890625
    ],
    "plans_name": "bs24",
    "transpose_backward": [
        0,
        1,
        2
    ],
    "transpose_forward": [
        0,
        1,
        2
    ]
}


to finetune:{
    "dataset_name": "finetune",
    "plans_name": "finetunennunet",
    "original_median_spacing_after_transp": [
        3.0,
        1.1875,
        1.1875
    ],
    "original_median_shape_after_transp": [
        72,
        258,
        318
    ],
    "image_reader_writer": "SimpleITKIO",
    "transpose_forward": [
        0,
        1,
        2
    ],
    "transpose_backward": [
        0,
        1,
        2
    ],
    "configurations": {
        "2d": {
            "data_identifier": "nnUNetPlans_2d",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 90,
            "patch_size": [
                320,
                320
            ],
            "median_image_size_in_voxels": [
                258.0,
                318.0
            ],
            "spacing": [
                1.1875,
                1.1875
            ],
            "normalization_schemes": [
                "ZScoreNormalization"
            ],
            "use_mask_for_norm": [
                false
            ],
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": false,
                "order": 3,
                "order_z": 0,
                "force_separate_z": null
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": true,
                "order": 1,
                "order_z": 0,
                "force_separate_z": null
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": false,
                "order": 1,
                "order_z": 0,
                "force_separate_z": null
            },
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                "arch_kwargs": {
                    "n_stages": 7,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        512,
                        512,
                        512
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv2d",
                    "kernel_sizes": [
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            2,
                            2
                        ]
                    ],
                    "n_blocks_per_stage": [
                        1,
                        3,
                        4,
                        6,
                        6,
                        6,
                        6
                    ],
                    "n_conv_per_stage_decoder": [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1
                    ],
                    "conv_bias": true,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": true
                    },
                    "dropout_op": null,
                    "dropout_op_kwargs": null,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": true
                    }
                },
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ]
            },
            "batch_dice": true
        },
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2,
            "patch_size": [
                64,
                256,
                320
            ],
            "median_image_size_in_voxels": [
                72.0,
                258.0,
                318.0
            ],
            "spacing": [
                3.0,
                1.1875,
                1.1875
            ],
            "normalization_schemes": [
                "ZScoreNormalization"
            ],
            "use_mask_for_norm": [
                false
            ],
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": false,
                "order": 3,
                "order_z": 0,
                "force_separate_z": null
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": true,
                "order": 1,
                "order_z": 0,
                "force_separate_z": null
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": false,
                "order": 1,
                "order_z": 0,
                "force_separate_z": null
            },
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                "arch_kwargs": {
                    "n_stages": 7,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320,
                        320
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [
                        [
                            1,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            1,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ]
                    ],
                    "n_blocks_per_stage": [
                        1,
                        3,
                        4,
                        6,
                        6,
                        6,
                        6
                    ],
                    "n_conv_per_stage_decoder": [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1
                    ],
                    "conv_bias": true,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": true
                    },
                    "dropout_op": null,
                    "dropout_op_kwargs": null,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": true
                    }
                },
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ]
            },
            "batch_dice": false
        }
    },
    "experiment_planner_used": "nnUNetPlannerResEncL",
    "label_manager": "LabelManager",
    "foreground_intensity_properties_per_channel": {
        "0": {
            "max": 1104.0,
            "mean": 244.73789978027344,
            "median": 226.0,
            "min": 0.0,
            "percentile_00_5": 45.0,
            "percentile_99_5": 618.0,
            "std": 115.88118743896484
        }
    }
}



"""
Evaluate 3D segmentation performance for all subjects in pred_dir,
using only .mha and .nii.gz files. This script:
 - Loads 3D masks and extracts voxel spacing.
 - Ensures prediction masks are binary (0 and 1).
 - If a prediction mask is uniform (all zeros or all ones), all metrics are set to the lowest value possible.
 - Computes surface-based metrics (Dice, Surface Dice at 5mm, Robust Hausdorff95, MASD).
 - Computes tumor volumes and later aggregates metrics (mean for most and RMSE for volumes).
"""

import os
import json
import shutil
import os
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from surface_distance import metrics as surface_metrics


ALLOWED_EXTENSIONS = [".mha", ".nii.gz"]
panther_msg = r"""\n
<Computing PANTHER Evaluation Metrics>
                  /)-._
                 Y. ' _]
          ,.._   |`--"=
         /    "-/   \\
/)      |   |_     `\|___
\:::::::\___/_\__\_______\\
"""
panther_msg2 = r"""\n
  _____________________________
  < PANTHER Evaluation Done! >
  -----------------------------
"""


def load_mask(file_path):
    """
    Loads a 3D mask from a file using SimpleITK.
    Allowed extensions: .mha, .nii.gz, (also .nii, .mhd if needed).
    Returns:
      mask: a numpy array representation of the image.
      spacing: a tuple with the voxel spacing (in mm).
    Raises an error if the file is not one of the allowed types or if the image is not 3D.
    """
    if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"Only {ALLOWED_EXTENSIONS} files are allowed. Got: {file_path}")
    image = sitk.ReadImage(file_path)
    mask = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()  # e.g., (1.0, 1.0, 1.0)
    if mask.ndim != 3:
        raise ValueError(
            f"Mask from {file_path} is not 3D (found shape: {mask.shape}).")
    return mask, spacing


def find_file(directory, subject, allowed_extensions=ALLOWED_EXTENSIONS):
    """
    Given a directory and a subject ID, returns the file path if a file with
    subject+extension exists, checking the allowed extensions.
    """
    for ext in allowed_extensions:
        file_path = os.path.join(directory, subject + ext)
        if os.path.exists(file_path):
            return file_path
    return None


def evaluate_segmentation_performance(pred_dir, gt_dir, subject_list=None, verbose=False):
    """
    Evaluates segmentation metrics for all subjects.
    - pred_dir: Directory containing prediction files (.mha or .nii.gz).
    - gt_dir: Directory containing ground truth files (.mha or .nii.gz).
    - subject_list: Either a list of subject IDs or a JSON file (with "subject_list" key).
    - verbose: If True, prints per-subject metrics.

    Returns a dictionary with per-subject metrics and aggregated metrics.
    """

    # Collect nnUNet_results fold_1 to fold_4 under fold_all in same directory to process
    results_main_dir = pred_dir
    pred_dir = os.path.join(pred_dir, "fold_all")

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    file_copy_count = 0
    folders_to_scan = [f"fold_{i}/validation" for i in range(5)]

    for fold_name in folders_to_scan:
        source_dir = os.path.join(results_main_dir,fold_name)

        # Check if the source directory exists before trying to scan it
        if not os.path.isdir(source_dir):
            print(f"Skipping: Directory '{source_dir}' not found.")
            continue # Move to the next folder

        print(f"--- Searching in '{source_dir}' ---")

        # 3. Find all files ending in .nii.gz using Path.glob()
        nii_files = list(Path(source_dir).glob("*.nii.gz"))

        if not nii_files:
            print("  No .nii.gz files found.")
            continue

        for nii_file_path in nii_files:
            # Define the full path for the destination file
            destination_path = os.path.join(pred_dir, nii_file_path.name)

            print(f"  -> Copying '{nii_file_path.name}'")
            shutil.copy(nii_file_path, destination_path)
            file_copy_count += 1


    # Load subject list from JSON file if subject_list is a filename/path.
    if isinstance(subject_list, (str, Path)):
        with open(subject_list, "r") as fp:
            subject_list = json.load(fp)["subject_list"]
    # If not provided, list all subject IDs from pred_dir (remove extension appropriately)
    if subject_list is None:
        subject_set = set()
        for f in os.listdir(pred_dir):
            f_lower = f.lower()
            for ext in ALLOWED_EXTENSIONS:
                if f_lower.endswith(ext):
                    subject_set.add(f[:-len(ext)])
                    break
        subject_list = sorted(subject_set)

    metrics_list = []
    for subj in subject_list:
        pred_file = find_file(pred_dir, subj)
        gt_file = find_file(gt_dir, subj)

        if pred_file is None:
            if verbose:
                print(f"Prediction file not found for subject {subj}")
            continue
        if gt_file is None:
            if verbose:
                print(f"Ground truth file not found for subject {subj}")
            continue

        try:
            mask_pred, spacing_pred = load_mask(pred_file)
            mask_gt, spacing_gt = load_mask(gt_file)
            print("Spacing pred is: ", spacing_pred)
            print("Spacing gt is: ", spacing_gt)
        except Exception as e:
            if verbose:
                print(f"Error loading subject (or mask and spacing) {subj}: {e}")
            continue

        # Check that the shapes match.
        if mask_gt.shape != mask_pred.shape:
            raise ValueError(
                f"Shape mismatch for subject {subj}: GT shape {mask_gt.shape} vs Pred shape {mask_pred.shape}")
        # Check that the voxel spacings match.
        if not np.allclose(spacing_gt, spacing_pred, rtol=0, atol=1e-4):
                raise ValueError(
                    f"Voxel spacing mismatch: GT spacing {spacing_gt} vs Pred spacing {spacing_pred}")


        # Ensure prediction mask is binary.
        """
        unique_vals = np.unique(mask_pred)
        if not (np.array_equal(unique_vals, [0]) or
                np.array_equal(unique_vals, [0, 1]) or
                np.array_equal(unique_vals, [1])):
            if len(unique_vals) > 2 and 0 in unique_vals:
                if verbose:
                    print(f"Subject {subj}: Remapping multi-class masks to binary for tumor (label 1).")
                    print(f"  Original GT unique values: {np.unique(mask_gt)}")
                    print(f"  Original Pred unique values: {np.unique(mask_pred)}")
                if verbose:
                    print(f"Prediction mask for subject {subj} has unique values {unique_vals}. Converting nonzero values to 1.")
                mask_pred = (mask_pred == 1).astype(np.uint8)
                #mask_gt = (mask_pred == 1).astype(np.uint8)
                if verbose:
                    print(f"  New GT unique values: {np.unique(mask_gt)}")
                    print(f"  New Pred unique values: {np.unique(mask_pred)}")
            else:
                raise ValueError(
                    f"Prediction mask for subject {subj} is not binary. Unique values: {unique_vals}")
        else:
            mask_pred = mask_pred.astype(np.uint8)
        """
        # Converting anything other than tumor:1 to background:0
        mask_pred = (mask_pred == 1).astype(np.uint8).astype(bool)
        mask_gt = (mask_gt == 1).astype(np.uint8).astype(bool)
        # Convert masks to boolean as required by the surface-distance library.
        mask_pred = mask_pred.astype(bool)
        mask_gt = mask_gt.astype(bool)

        
        # Check for uniform prediction (all zeros or all ones)
        if np.all(mask_pred == 0) or np.all(mask_pred == 1):
            if verbose:
                # GT non-empty but prediction empty or full of 1s→ complete miss.
                print(f"Subject {subj}: Prediction mask is uniform. Metrics set to 0.")
            # Compute max_distance to set distance metrics.
            max_distance = np.linalg.norm(
                np.array(mask_gt.shape) * np.array(spacing_gt))
            # Overlap-based metrics are 0; distances get the penalty.
            subj_metrics = {
                "subject": subj,
                "volumetric_dice": 0.0,
                "surface_dice": 0.0,
                "hausdorff95": max_distance,
                "masd": max_distance,
                "gt_volume": np.sum(mask_gt) * np.prod(spacing_gt),
                "pred_volume": 0.0,
                "time_score": 0.0
            }
            metrics_list.append(subj_metrics)
            continue

        # Compute surface-based metrics using the ground truth spacing.
        surface_distances = surface_metrics.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=spacing_gt)
        dice = surface_metrics.compute_dice_coefficient(mask_gt, mask_pred)
        surf_dice = surface_metrics.compute_surface_dice_at_tolerance(
            surface_distances, tolerance_mm=5)
        hausdorff95 = surface_metrics.compute_robust_hausdorff(
            surface_distances, percent=95)
        avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(
            surface_distances)
        masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0

        # Compute tumor volumes using the ground truth spacing.
        voxel_volume = np.prod(spacing_gt)
        gt_volume = np.sum(mask_gt) * voxel_volume
        pred_volume = np.sum(mask_pred) * voxel_volume

        subj_metrics = {
            "subject": subj,
            "volumetric_dice": dice,
            "surface_dice": surf_dice,
            "hausdorff95": hausdorff95,
            "masd": masd,
            "gt_volume": gt_volume,
            "pred_volume": pred_volume,
        }
        metrics_list.append(subj_metrics)
        if verbose:
            print(f"Subject: {subj}")
            print(f"  Volumetric Dice: {dice:.4f}")
            print(f"  Surface Dice (5mm): {surf_dice:.4f}")
            print(f"  Hausdorff95: {hausdorff95:.4f}")
            print(f"  MASD: {masd:.4f}")
            print(
                f"  GT Volume: {gt_volume:.2f} mm³, Pred Volume: {pred_volume:.2f} mm³")

    # Aggregate metrics across subjects.
    if len(metrics_list) == 0:
        raise RuntimeError("No subjects were processed successfully!")

    mean_dice = np.mean([m["volumetric_dice"] for m in metrics_list])
    mean_surf_dice = np.mean([m["surface_dice"] for m in metrics_list])
    mean_hausdorff95 = np.mean([m["hausdorff95"] for m in metrics_list])
    mean_masd = np.mean([m["masd"] for m in metrics_list])

    # For tumor volumes, compute RMSE.
    gt_volumes = np.array([m["gt_volume"] for m in metrics_list])
    pred_volumes = np.array([m["pred_volume"] for m in metrics_list])
    rmse_volume = np.sqrt(np.mean((pred_volumes - gt_volumes) ** 2))

    # Delete the fold_all folder after metrics are calculated to not take diskspace
    shutil.rmtree(pred_dir)

    aggregates = {
        "mean_volumetric_dice": mean_dice,
        "mean_surface_dice": mean_surf_dice,
        "mean_hausdorff95": mean_hausdorff95,
        "mean_masd": mean_masd,
        "tumor_burden_rmse": rmse_volume,
    }

    return {
        "per_subject": metrics_list,
        "aggregates": aggregates,
    }

    


if __name__ == "__main__":
    import argparse
    import json
            
    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation performance for .mha and .nii.gz masks")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction files (.mha or .nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth files (.mha or .nii.gz)")
    parser.add_argument("--subject_list", type=str, default=None,
                        help="Optional JSON file with {'subject_list': [...]}, or a comma-separated list of subject IDs")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional path to save the aggregated metrics as a JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    print(panther_msg)
    # Process subject list argument.
    subject_list = args.subject_list
    if subject_list is not None:
        if subject_list.endswith(".json"):
            with open(subject_list, "r") as fp:
                subject_list = json.load(fp)["subject_list"]
        else:
            subject_list = [s.strip() for s in subject_list.split(",")]

    results = evaluate_segmentation_performance(args.pred_dir, args.gt_dir,
                                                  subject_list=subject_list,
                                                  verbose=args.verbose)
    print("Evaluation Metrics:")
    print(json.dumps(results, indent=4))

        # Save the metrics JSON if a path is provided.
    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {args.save_path}")
    print(panther_msg2)

    
