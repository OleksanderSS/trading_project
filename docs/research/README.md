# DEAN-OS Research Dropzone

This folder is for operator-supplied local research materials that can enter the
review-only real-source normalization path.

Accepted file types:

- `.txt`
- `.md`
- `.markdown`
- `.html`
- `.htm`
- `.json`
- `.pdf`
- `.docx`

Boundary:

- Files placed here are not automatically production evidence.
- Normalization creates review artifacts only.
- No live fetch, external API call, claim extraction, event extraction, thesis,
  valuation, recommendation, learning write, paper trade, or live trade is
  authorized by this folder.
- Candidate ticker, sector, and topic tags are routing hints only until reviewed.

Inventory command:

```powershell
python run_agent_real_source_dropzone_inventory.py --dropzone docs\research --output-dir reports\dean_os\real_source_dropzone_inventory_current
```

First review-only command:

```powershell
python run_agent_real_source_normalized_packet.py docs\research\YOUR_FILE.md --source-type report --ticker AMD --sector semiconductors --tag semiconductor_supply_chain --output-dir reports\dean_os\real_source_normalized_packet_current
```

Validation command:

```powershell
python run_review_only_real_source_normalized_packet_validation_gate.py --input-json reports\dean_os\real_source_normalized_packet_current\latest.json --output-dir reports\dean_os\real_source_normalized_packet_validation_gate_current
```

Review the generated `latest.md` and `latest.json` artifacts before any
extraction contract or analyst workflow consumes the packet.
