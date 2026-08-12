# Grafana monitoring (module 05, lesson 12)

Grafana is the dedicated dashboarding tool used once the Streamlit
dashboard is no longer enough: it reads straight from the monitoring
Postgres (`postgres` service in docker-compose.yml) and shows response
speed, cost, relevance and user ratings in near real time.

Everything is provisioned as code — no clicking in the UI:

| Path | Purpose |
|------|---------|
| `provisioning/datasources/datasource.yml` | PostgreSQL data source pointing at the `course-assistant-pg` container |
| `provisioning/dashboards/dashboards.yml` | File provider that loads dashboards from `/var/lib/grafana/dashboards` (mapped to this folder) |
| `dashboards/course-assistant.json` | The dashboard: 7 panels, 30s auto-refresh, last-6h default range |

## Panels

| Panel | Type | Query highlights |
|-------|------|------------------|
| Recent conversations | Table | last 5 calls, `LIMIT 5`, newest first |
| Model usage | Bar chart | `COUNT(*) GROUP BY model` |
| Relevance distribution | Pie chart | judge verdicts from `feedback` (`source = 'judge'`) |
| User feedback | Gauge | thumbs up vs down sums from `feedback` (`source = 'user'`) |
| Response time | Time series | raw `response_time` per row |
| Token usage | Time series | `AVG(total_tokens)` bucketed by `$__timeGroup` |
| Cost | Time series | `SUM(cost)` bucketed, `cost > 0` |

Every panel filters on the selected time range with `$__timeFrom()` /
`$__timeTo()`, and aliases the time column as `time` so Grafana places
points on the x-axis correctly.

## Running

```bash
docker compose up -d postgres grafana
uv run python db_init.py      # once: create conversations + feedback tables
uv run python generate_data.py  # optional: stream synthetic data
```

Then open http://localhost:3000 (first login admin/admin) — the
dashboard "Course Assistant Monitoring" is already there.

Follows the llm-zoomcamp lesson "Grafana Dashboards":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/12-grafana.md
