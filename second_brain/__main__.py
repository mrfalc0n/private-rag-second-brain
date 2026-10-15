"""Command line: python -m second_brain {sync <source> [--full] | search "question" | status | login gdrive | folders [name]}"""

import argparse
import json
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="second_brain")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sync = sub.add_parser("sync", help="fetch and index one source")
    p_sync.add_argument("source", choices=["notion", "gdrive", "files", "dummy"])
    p_sync.add_argument("--full", action="store_true", help="ignore the saved cursor")
    p_search = sub.add_parser("search", help="vector search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    sub.add_parser("status", help="show sync_state")
    p_login = sub.add_parser("login", help="one-time browser login for a Google source")
    p_login.add_argument("source", choices=["gdrive"])
    p_folders = sub.add_parser("folders", help="find Google Drive folder IDs by name")
    p_folders.add_argument("name", nargs="?")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "sync":
        from second_brain import sync
        if args.source == "notion":
            from second_brain.connectors.notion import NotionConnector
            connector = NotionConnector()
        elif args.source == "gdrive":
            from second_brain.connectors.gdrive import GDriveConnector
            connector = GDriveConnector()
        elif args.source == "files":
            from second_brain.connectors.files import FilesConnector
            connector = FilesConnector()
        else:
            from second_brain.connectors.dummy import DummyConnector
            connector = DummyConnector()
        print(json.dumps(sync.run(connector, full=args.full), indent=2, default=str))
    elif args.command == "search":
        from second_brain.retrieval import search
        for hit in search(args.query, k=args.k):
            print(f"{hit['score']:.3f}  {hit['title']}  [{hit['source']}]  {hit['url']}")
            print("       " + hit["snippet"].replace("\n", " ")[:160])
    elif args.command == "login":
        from second_brain.connectors.gdrive import credentials
        credentials(interactive=True)
        print("Google Drive login saved.")
    elif args.command == "folders":
        from second_brain.connectors.gdrive import find_folders
        for f in find_folders(args.name):
            print(f"{f['id']}  {f['name']}  (modified {f['modifiedTime'][:10]})")
    else:
        from second_brain import store
        store.ensure_schema()
        print(json.dumps(store.status(), indent=2, default=str))


if __name__ == "__main__":
    main()
