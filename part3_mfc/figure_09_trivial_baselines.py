"""Physica D Fig. 7 and Fig. S10 (historical pipeline slot 09)."""
if __package__:
    from .current_controller import figure_main
else:
    from current_controller import figure_main
def main():return figure_main(9)
if __name__=='__main__':raise SystemExit(main())
