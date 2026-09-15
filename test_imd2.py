from tools.imd_service import get_imd_coastal_bulletin

for name, lat, lon, hint in [
    ("Goa",      15.3004543, 74.0855134, "Goa"),
    ("Mumbai",   19.0760,    72.8777,    "Maharashtra"),
    ("Chennai",  13.0827,    80.2707,    "Tamil Nadu"),
    ("Kochi",     9.9312,    76.2673,    "Kerala"),
]:
    r = get_imd_coastal_bulletin(lat, lon, state_hint=hint)
    print("=" * 60)
    print(f"{name}  (bulletin id={r.get('bulletin_id', '?')})")
    print(f"  status:        {r.get('status')}")
    print(f"  centre:        {r.get('centre')}")
    print(f"  region:        {r.get('region')}")
    print(f"  issued_at:     {r.get('issued_at')}")
    print(f"  wind:          {r.get('wind')}")
    print(f"  weather:       {r.get('weather')}")
    print(f"  visibility:    {r.get('visibility')}")
    print(f"  sea_condition: {r.get('sea_condition')}")
    print(f"  port_signal:   {r.get('port_signal')}")
    print(f"  port_active:   {r.get('port_signal_active')}")
    print(f"  storm_surge:   {r.get('storm_surge_warning')}")
    print(f"  surge_active:  {r.get('storm_surge_active')}")