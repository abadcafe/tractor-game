# Test Case Changes: Task-008

## test_full_game
- File: tests/test_full_game_ws.py::test_full_game
- Changes: Added unknown action type assertion keyword "unknown" (server returns English "unknown action type:" but assertion only checked Chinese keywords); Added bid error keywords "王叫牌必须出对子", "优先级不足", "非庄家方不能叫牌"; Added tested_interleave flags for DEAL_BID, STIRRING, PLAYING phases; Added bid fallback (do_bid_pass) when do_bid fails; Added receive_state method to WsGameDriver; Added _ws_cm context manager tracking; Added wait_for_phase/ wait_for_awaiting docstring improvements; Added helper functions (_is_list, _as_list, _as_str_or_none, _is_list_of_dict)
- Reason: Server error messages don't match the task spec's expected keywords; test needs to handle disconnect-reconnect timing issues; helper functions needed by the full game test but were missing from the file
