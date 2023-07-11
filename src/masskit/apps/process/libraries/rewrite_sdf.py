import hydra
import rich.progress
import re

"""
reformat pre v2000 sdf files into those usable by rdkit by appending missing M END lines.
Turns any latin-1 characters into the unicode expected by rdkit


"""

@hydra.main(config_path="conf", config_name="config_rewrite_sdf", version_base=None)
def rewrite_sdf_app(config):

    # match connectivity block line
    connectivity_block = re.compile(r'^\s{0,2}\d{1,3}\s{0,2}\d{1,3}\s{1,2}\d+\s{1,2}(\d{1,3}|\s)\s{1,2}(\d{1,3}|\s)\s{1,2}(\d{1,3}|\s)\s{0,2}(\d{1,3}|\s)?\s*$')
                        
    with rich.progress.open(config.input.file.name, 'rt', encoding=config.input.file.encoding, 
                            description=f"{config.input.file.name} -> {config.output.file.name}") as fin:
        with open(config.output.file.name, 'w') as fout:
            previous_line = ""
            first_line = True
            in_m_block = False
            no_m_end = True
            for line in fin:
                if not first_line:
                    if connectivity_block.match(previous_line) and \
                    not connectivity_block.match(line):
                        in_m_block = True
                        no_m_end = True
                    if in_m_block and line.strip() == 'M  END':
                        no_m_end = False
                    if in_m_block and (line.startswith("> ") or line.startswith('$$$$')):
                        in_m_block = False
                        if no_m_end:
                            fout.write('M  END\n')
                first_line = False
                fout.write(line)
                previous_line = line

if __name__ == "__main__":
    rewrite_sdf_app()