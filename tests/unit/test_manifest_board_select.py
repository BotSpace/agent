from agent.flow.manifest_compiler.compiler import compile_node_model


def test_board_select_compiles_board_and_column_as_strict_strings():
    manifest = {
        "type": "KanbanCardNode",
        "content": [
            {
                "type": "board-select",
                "key": "boardId",
                "columnKey": "columnId",
                "label": "Board va ustun",
            }
        ],
        "defaults": {"boardId": "", "columnId": ""},
    }

    model = compile_node_model(manifest)
    value = model(boardId="board-uuid", columnId="column-uuid")

    assert value.boardId == "board-uuid"
    assert value.columnId == "column-uuid"
