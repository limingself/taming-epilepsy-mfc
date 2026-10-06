"""Current RMS-matched baselines for the same Full700 controller."""
if __package__:
    from .current_controller import figure_main
else:
    from current_controller import figure_main
def main():return figure_main(9)
if __name__=='__main__':raise SystemExit(main())
