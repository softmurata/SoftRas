
function main(){
    $start = 0
    $finish = 777
    for ($i=$start; $i -lt $finish; $i++){
        python multiview_script.py --raw_dataset_dir './RawDataset144/' --mesh_number $i
    }
}

main
