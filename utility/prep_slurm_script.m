function prep_slurm_script(slurm_folder, slurm_name, param_name,numprocs)

%%%%%%%%%%%%%%%%%%%%% create the slurm script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% outdir = [ pwd filesep 'amicaouttmp' filesep ];

% if slurm_folder(1) == '.'
%     slurm_folder = [pwd slurm_folder(2:end)];
% end
% if slurm_folder(end) ~= filesep
%     slurm_folder(end+1) = filesep;
% end
% system(['mkdir ' slurm_folder]);

fid = fopen([slurm_folder slurm_name],'w');
if fid < 1
    errordlg(['cannot create file in ' slurm_folder],'Bad Input','modal');
    return;
end

%%%%%%%%%%% generate param file for Comet %%%%%%%%%%%%
file_param = sprintf('/home/shhsu/amica/param/emotion/%s.param',param_name);

fprintf(fid,'#!/bin/bash\n');
fprintf(fid,'#BATCH --job-name="amica_mpi_emotion"\n');
fprintf(fid,'#SBATCH --output="%s.out"\n',param_name);
fprintf(fid,'#SBATCH --partition=compute\n');
fprintf(fid,'#SBATCH --nodes=%d\n',numprocs);
fprintf(fid,'#SBATCH --ntasks-per-node=24\n');
fprintf(fid,'#SBATCH --export=ALL\n');
fprintf(fid,'#SBATCH -t 48:00:00\n\n');

fprintf(fid,'#SET the number of openmp threads\n');
fprintf(fid,'export OMP_NUM_THREADS=24\n');
fprintf(fid,'export MV2_ENABLE_AFFINITY=0\n\n');

fprintf(fid,'#Run the job\n');
fprintf(fid,'cd /home/shhsu/amica/\n');
fprintf(fid,'ibrun --npernode 1 ./amica15c %s',file_param);

fclose(fid);
