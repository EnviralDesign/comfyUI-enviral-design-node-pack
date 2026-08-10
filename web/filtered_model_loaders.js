import { app } from "/scripts/app.js";

const FILTERED_LOADERS = {
    EnviralLoadCheckpointFiltered: {
        allFolders: "All Checkpoints",
        selectionInput: "ckpt_name",
    },
    EnviralLoadDiffusionModelFiltered: {
        allFolders: "All Diffusion Models",
        selectionInput: "unet_name",
    },
    EnviralLoadTextEncoderFiltered: {
        allFolders: "All Text Encoders",
        selectionInput: "clip_name",
    },
};

function normalizePath(value) {
    return String(value ?? "").trim().replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getInputOptions(nodeData, name) {
    const input = nodeData?.input?.required?.[name] ?? nodeData?.input?.optional?.[name];
    if (!Array.isArray(input)) {
        return null;
    }
    if (Array.isArray(input[0])) {
        return input[0];
    }
    return Array.isArray(input[1]?.options) ? input[1].options : null;
}

function getConnectedAllowList(node) {
    const slot = node.findInputSlot?.("allow_list");
    const linkId = slot === undefined || slot < 0 ? null : node.inputs?.[slot]?.link;
    const link = linkId == null ? null : app.graph?.links?.[linkId];
    const sourceNode = link ? app.graph?.getNodeById?.(link.origin_id) : null;
    const sourceWidget = sourceNode?.widgets?.find((widget) => widget.name === "value");
    return { sourceWidget, value: sourceWidget?.value ?? "" };
}

function parseAllowList(value) {
    return new Set(
        String(value ?? "")
            .split(/\r?\n/)
            .map((line) => normalizePath(line).toLowerCase())
            .filter(Boolean),
    );
}

function isInFolder(fileName, folder, allFolders) {
    const normalizedFolder = normalizePath(folder).toLowerCase();
    if (normalizedFolder === normalizePath(allFolders).toLowerCase()) {
        return true;
    }
    const normalizedName = normalizePath(fileName).toLowerCase();
    return normalizedFolder !== "" && normalizedName.startsWith(`${normalizedFolder}/`);
}

function getDisplayName(fileName, folder, allFolders) {
    const name = String(fileName ?? "");
    const normalizedFolder = normalizePath(folder);
    if (
        normalizedFolder === "" ||
        normalizedFolder.toLowerCase() === normalizePath(allFolders).toLowerCase()
    ) {
        return name;
    }

    const normalizedName = normalizePath(name);
    const prefix = `${normalizedFolder}/`;
    if (!normalizedName.toLowerCase().startsWith(prefix.toLowerCase())) {
        return name;
    }
    return normalizedName.slice(prefix.length);
}

function isAllowed(fileName, folder, allFolders, allowed) {
    if (allowed.size === 0) {
        return true;
    }
    const fullName = normalizePath(fileName).toLowerCase();
    const displayName = normalizePath(getDisplayName(fileName, folder, allFolders)).toLowerCase();
    return allowed.has(fullName) || allowed.has(displayName);
}

function configureFilter(node, nodeName, config) {
    const folderWidget = getWidget(node, "folder");
    const selectionWidget = getWidget(node, config.selectionInput);
    if (!folderWidget || !selectionWidget || !Array.isArray(selectionWidget.options?.values)) {
        return;
    }

    let allFiles = [...selectionWidget.options.values];
    selectionWidget.options.getOptionLabel = (fileName) =>
        getDisplayName(fileName, folderWidget.value, config.allFolders);

    function filteredFiles() {
        const allowed = parseAllowList(getConnectedAllowList(node).value);
        return allFiles.filter(
            (fileName) =>
                isInFolder(fileName, folderWidget.value, config.allFolders) &&
                isAllowed(fileName, folderWidget.value, config.allFolders, allowed),
        );
    }

    function syncSelection() {
        const filtered = filteredFiles();
        if (!filtered.includes(selectionWidget.value)) {
            selectionWidget.value = filtered[0] ?? "";
        }
        node.setDirtyCanvas?.(true, true);
    }

    Object.defineProperty(selectionWidget.options, "values", {
        configurable: true,
        enumerable: true,
        get: filteredFiles,
        set(values) {
            if (Array.isArray(values)) {
                allFiles = [...values];
                syncSelection();
            }
        },
    });

    function sync() {
        const { sourceWidget } = getConnectedAllowList(node);
        if (node.enviralFilterSource && node.enviralFilterSource !== sourceWidget) {
            node.enviralFilterSource.enviralFilterTargets?.delete(node);
        }
        node.enviralFilterSource = sourceWidget;
        if (sourceWidget) {
            sourceWidget.enviralFilterTargets ??= new Set();
            sourceWidget.enviralFilterTargets.add(node);
            if (!sourceWidget.enviralFilterWrapped) {
                const originalCallback = sourceWidget.callback;
                sourceWidget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    for (const target of sourceWidget.enviralFilterTargets) {
                        target.enviralSyncFileFilter?.();
                    }
                    return result;
                };
                sourceWidget.enviralFilterWrapped = true;
            }
        }
        syncSelection();
    }

    const originalFolderCallback = folderWidget.callback;
    folderWidget.callback = function () {
        const result = originalFolderCallback?.apply(this, arguments);
        sync();
        return result;
    };

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        queueMicrotask(() => this.enviralSyncFileFilter?.());
        return result;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        this.enviralFilterSource?.enviralFilterTargets?.delete(this);
        return onRemoved?.apply(this, arguments);
    };

    const refreshComboInNode = node.refreshComboInNode;
    node.refreshComboInNode = function (nodeDefs) {
        const result = refreshComboInNode?.apply(this, arguments);
        const nodeData = nodeDefs?.[nodeName];
        const folders = getInputOptions(nodeData, "folder");
        const files = getInputOptions(nodeData, config.selectionInput);
        if (folders) {
            folderWidget.options.values = folders;
            if (!folders.includes(folderWidget.value)) {
                folderWidget.value = folders.includes(config.allFolders)
                    ? config.allFolders
                    : (folders[0] ?? config.allFolders);
            }
        }
        if (files) {
            selectionWidget.options.values = files;
        }
        this.enviralSyncFileFilter?.();
        return result;
    };

    node.enviralSyncFileFilter = sync;
    queueMicrotask(sync);
}

app.registerExtension({
    name: "EnviralDesign.FilteredModelLoaders",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = FILTERED_LOADERS[nodeData.name];
        if (!config) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            configureFilter(this, nodeData.name, config);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            queueMicrotask(() => this.enviralSyncFileFilter?.());
            return result;
        };
    },
});
