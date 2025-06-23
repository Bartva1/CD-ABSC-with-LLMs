from utilities import parse_experiment_args, generate_info
import pprint

if __name__ == "__main__":

    source_domains, target_domains, demos, models, shot_infos, indices = parse_experiment_args()
    
    test_info = generate_info(
        source_domains=source_domains,
        target_domains=target_domains,
        demos=demos,
        models=models,
        shot_infos=shot_infos,
        indices=indices
    )

    print("Generated test_info combinations:\n")
    print("format is 'source', 'target', 'demo', 'model', 'shot_info'")
    pprint.pprint(test_info)
    print(f"\nTotal combinations: {len(test_info)}")
