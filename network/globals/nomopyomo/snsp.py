# def add_snsp_constraint_tyndp(net: pypsa.Network, snsp_share: float):
#     """
#     Add system non-synchronous generation share constraint to the model.
#
#     Parameters
#     ----------
#     net: pypsa.Network
#         A PyPSA Network instance with buses associated to regions
#     snsp_share: float
#         Share of system non-synchronous generation.
#
#     """
#     # TODO: DC to be included, however the constraint should then be imposed on a nodal basis
#     snapshots = net.snapshots
#
#     nonsync_gen_types = 'wind|pv'
#     nonsync_storage_types = 'Li-ion'
#
#     store_links = net.links[net.links.index.str.contains('link')]
#
#     gens_p = get_var(net, 'Generator', 'p')
#     storageunit_p = get_var(net, 'StorageUnit', 'p_dispatch')
#     link_p = get_var(net, 'Link', 'p')
#
#     for s in snapshots:
#
#         gens_p_s = gens_p.loc[s, :]
#         storageunit_p_s = storageunit_p.loc[s, :]
#         link_p_s = link_p.loc[s, :]
#
#         nonsync_gen_ids = net.generators.index[(net.generators.type.str.contains(nonsync_gen_types))]
#         nonsyncgen_p = gens_p_s.loc[nonsync_gen_ids]
#
#         nonsync_storageunit_ids = \
#                               net.storage_units.index[(net.storage_units.type.str.contains(nonsync_storage_types))]
#         nonsyncstorageunit_p = storageunit_p_s.loc[nonsync_storageunit_ids]
#
#         nonsync_store_ids = store_links.index[(store_links.index.str.contains(nonsync_storage_types)) &
#                                               (store_links.index.str.contains('to AC'))]
#         nonsyncstore_p = link_p_s.loc[nonsync_store_ids]
#
#         lhs_gen_nonsync = linexpr((1., nonsyncgen_p)).values
#         lhs_storun_nonsync = linexpr((1., nonsyncstorageunit_p)).values
#         lhs_store_nonsycn = linexpr((1., nonsyncstore_p)).values
#         lhs_gens = linexpr((-snsp_share, gens_p_s)).values
#         lhs_storage = linexpr((-snsp_share, storageunit_p_s)).values
#
#         lhs = np.concatenate((lhs_gen_nonsync, lhs_storun_nonsync, lhs_store_nonsycn, lhs_gens, lhs_storage))
#
#         define_constraints(net, lhs, '<=', 0., 'snsp_constraint', s)
