def argIsNone(args):
    if isinstance(args, list):
        assert len(args) == 1
        if args[0] == None :
            return True
        elif isinstance(args[0], str):
            if args[0].lower() == 'none':
                return True
        else:
            return False
    else:
        if args == None:
            return True
        elif isinstance(args, str):
            if args.lower == "none":
                return True
        else:
            return False