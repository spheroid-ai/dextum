# Sensorised Aero model v2

- [Original XML](kit_b_friction_matched.original.xml): exact accepted model,
  including historical absolute asset paths.
- [Portable XML](kit_b_friction_matched.xml): only asset paths changed to local
  `assets/` paths.
- [Referenced assets](assets/) and [sensor geometry](geometry.json).

Accepted original SHA-256:

```text
a2e0c670506cb69267bac11e80fa24c467a782cca3bedb901290adc64444c924
```

The verifier compares both XML trees after normalising only asset paths.
No contact parameter, body, geometry, sensor or solver value changes in the
portable copy. Its hash necessarily differs from the original.

V2 preserves the sensorised geometry and matches pad friction to the hand's
geoms. The interface has ten pads and forty cell sites; compound palm geometry
means the total collider count is not ten. Stiffness remains a placeholder;
primitive soft contacts do not reproduce conformal contact area.

The training adapter sets the integration timestep and adds randomisation,
sensors, reset logic and task scoring. This standalone asset is not the full
training protocol.

The Aero base model and meshes retain the TetherIA Apache-2.0
[licence](UPSTREAM_AERO_LICENSE). Original DEXTUM model changes use the software
licence; original kit meshes follow [LICENSE-DATA.md](../../../LICENSE-DATA.md).
No physical hardware performance or complete manufacturing release is implied.
