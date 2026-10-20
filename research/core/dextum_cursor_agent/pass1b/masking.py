"""Packet-level masks shared by actor and reflex; reward truth is never masked."""
import jax.numpy as j
GROUPS=('all','no_shear','no_pvdf_slip','no_barometric')
def mask_packet(packet,group):
    if group not in GROUPS:raise ValueError(group)
    if group=='no_shear':
        optical=packet[:60].reshape(20,3).at[:,1:].set(0)
        return packet.at[:60].set(optical.ravel()).at[60:100].set(0)
    if group=='no_pvdf_slip':return packet.at[135:160].set(0)
    if group=='no_barometric':return packet.at[100:120].set(0).at[180:200].set(0)
    return packet
