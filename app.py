from crypto.mlp_experiences.mlp_optimization import optimize_mlp_h1, show_val_losses_h1, h1_metrics, h1_analyze

if __name__ == '__main__':
    optimize_mlp_h1()
    h1_analyze()
    # h1_metrics()
    show_val_losses_h1()
