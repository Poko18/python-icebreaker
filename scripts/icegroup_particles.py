import argparse
import collections
from multiprocessing import Pool
import numpy as np
import starfile
from icebreaker import icebreaker_icegroups_multi

def load_star_file(star_file):
    """Load the star file and extract particle data."""
    df = starfile.read(star_file)
    particles = df["particles"]
    particles["ibIceGroup"] = -1

    if "rlnMicrographName" in particles.columns:
        micrographs_used = particles["rlnMicrographName"]
    else:
        print("No micrograph name present, exiting")
        exit()

    return df, particles, micrographs_used

def get_unique_micrographs(micrographs_used):
    """Get unique micrograph names and create a dictionary mapping each micrograph to its corresponding particle indices."""
    micrographs_unique = list(set(micrographs_used))
    micrographs_unique.sort()

    mic_coord = collections.OrderedDict()
    for mic in micrographs_unique:
        mic_coord[mic] = [i for i, e in enumerate(micrographs_used) if e == mic]

    return micrographs_unique, mic_coord

def process_particle_icethickness(args, mic):
    """Process ice thickness for a given micrograph."""
    img = icebreaker_icegroups_multi.load_img(mic)
    seg_img = icebreaker_icegroups_multi.ice_grouper(img, args.x_patches, args.y_patches, args.num_of_segments)

    ice_group = {}
    for part_ind in mic_coord[mic]:
        x1 = int(np.floor(particles["rlnCoordinateX"][part_ind]))
        y1 = int(np.floor(particles["rlnCoordinateY"][part_ind]))

        if seg_img is not None and np.isfinite(seg_img[y1][x1]):
            ice_group[part_ind] = int(seg_img[y1][x1] * 10000)

    return ice_group

def apply_ice_group_to_particles(result, particles):
    """Apply ice group labels to particle data."""
    for part_ind, ice_group in result.items():
        particles.at[part_ind, 'ibIceGroup'] = ice_group

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Assign ice group labels to particles based on image segmentation.")
    parser.add_argument("starfile", help="Input star file containing particle information.")
    parser.add_argument("--cpus", type=int, default=12, help="Number of CPU cores to use for parallel processing.")
    parser.add_argument("--x_patches", type=int, default=40, help="Number of patches in the x-direction for ice grouper.")
    parser.add_argument("--y_patches", type=int, default=40, help="Number of patches in the y-direction for ice grouper.")
    parser.add_argument("--num_of_segments", type=int, default=16, help="Number of segments for ice grouper.")
    parser.add_argument("--output_star", default="extract_particles_icegroups.star", help="Output star file name.")
    args = parser.parse_args()

    # Load star file and extract particle data
    df, particles, micrographs_used = load_star_file(args.starfile)

    # Get unique micrographs and create a dictionary mapping micrographs to particle indices
    micrographs_unique, mic_coord = get_unique_micrographs(micrographs_used)

    # Process ice thickness in parallel
    with Pool(args.cpus) as p:
        results = p.starmap(process_particle_icethickness, [(args, mic) for mic in micrographs_unique])

    # Apply ice group labels to particle data
    for result in results:
        apply_ice_group_to_particles(result, particles)

    # Save the updated particle data to a new star file
    df["particles"] = particles
    starfile.write(df, args.output_star)
