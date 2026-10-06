"""Current selected-update700 four-law all36 contact figure."""
if __package__:
    from .current_controller import figure_main
else:
    from current_controller import figure_main
def main():return figure_main(8)
if __name__=='__main__':raise SystemExit(main())
