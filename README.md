1. 运行 examples/voxceleb/v2/run.sh中的stage 1, local/prepare_data.sh中stage 4保存voxceleb2的路径改成你的地址
2. 运行 examples/voxceleb/v2/run.sh中的stage 2,data_type='raw'
3. 准备Augmentation:生成RIRS_NOISES/file_list和musan/file_list,每个file_list中列出RIRS和MUSAN中每个样本的路径,可参考该目录中的rirs_lis和musan_list文件
4. 修改examples/voxceleb/v2/run.sh中的stage 3的参数--reverb_data和--noise_data,--pho_path不用管没啥用
5. 在wespeaker/dataset/processor.py中第231行修改每个sample的音素文件路径
6. 运行run.sh stage 3
