from ..cli import CliSubcommand
from ..mutate import NodeMutator
from argparse import _SubParsersAction


class MutateCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "mutate",
            "Mutate strategies to gain insight!",
        )
        p = self.parser
        p.add_argument("cfr_file", help="PioSOLVER file to open")
        p.add_argument(
            "node_id",
            help='The node id to investigate (In PioViewer, Right-click the board and select "Copy node id")',
        )
        p.add_argument(
            "--save",
            action="store_true",
            help="Store output with np.save(...) to saved_deltas/ directory",
        )

    def run(self, args) -> int:
        mutator = NodeMutator(cfr_file=args.cfr_file, node=args.node_id)
        if args.save:
            mutator.save = True
        try:
            mutator.run()
            return 0
        except RuntimeError as e:
            print(e)
            return 1
