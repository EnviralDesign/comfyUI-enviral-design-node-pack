import { app } from "/scripts/app.js";

const NODE_NAME = "EnviralLoadLoraFiltered";
const ALL_FOLDERS = "All LoRAs";
const MAX_BANKS = 5;
const BANK_WIDGET_NAMES = ["lora_name", "strength_model", "strength_clip"];

function normalizePath(value) {
    return String(value ?? "").trim().replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
}

function isInFolder(loraName, folder) {
    const normalizedFolder = normalizePath(folder).toLowerCase();
    if (normalizedFolder === normalizePath(ALL_FOLDERS).toLowerCase()) {
        return true;
    }
    const normalizedName = normalizePath(loraName).toLowerCase();
    return normalizedFolder !== "" && normalizedName.startsWith(`${normalizedFolder}/`);
}

function getLoraDisplayName(loraName, folder) {
    const name = String(loraName ?? "");
    const normalizedFolder = normalizePath(folder);
    if (
        normalizedFolder === "" ||
        normalizedFolder.toLowerCase() === normalizePath(ALL_FOLDERS).toLowerCase()
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

function parseAllowList(value) {
    return new Set(
        String(value ?? "")
            .split(/\r?\n/)
            .map((line) => normalizePath(line).toLowerCase())
            .filter(Boolean),
    );
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getBankWidgetName(name, bankIndex) {
    return bankIndex === 1 ? name : `${name}_${bankIndex}`;
}

function getBankCount(node) {
    const count = Number.parseInt(getWidget(node, "bank_count")?.value, 10);
    if (!Number.isFinite(count)) {
        return 1;
    }
    return Math.min(MAX_BANKS, Math.max(1, count));
}

function getConnectedAllowList(node) {
    const slot = node.findInputSlot?.("allow_list");
    const linkId = slot === undefined || slot < 0 ? null : node.inputs?.[slot]?.link;
    const link = linkId == null ? null : app.graph?.links?.[linkId];
    const sourceNode = link ? app.graph?.getNodeById?.(link.origin_id) : null;
    const sourceWidget = sourceNode?.widgets?.find((widget) => widget.name === "value");
    return { sourceWidget, value: sourceWidget?.value ?? "" };
}

function isAllowed(loraName, folder, allowed) {
    if (allowed.size === 0) {
        return true;
    }
    const fullName = normalizePath(loraName).toLowerCase();
    const displayName = normalizePath(getLoraDisplayName(loraName, folder)).toLowerCase();
    return allowed.has(fullName) || allowed.has(displayName);
}

function configureLoraFilter(node, folderWidget, loraWidget) {
    if (!Array.isArray(loraWidget.options?.values)) {
        return;
    }

    let allLoras = [...loraWidget.options.values];
    loraWidget.options.getOptionLabel = (loraName) =>
        getLoraDisplayName(loraName, folderWidget.value);

    const valuesDescriptor = {
        configurable: true,
        enumerable: true,
        get() {
            const allowed = parseAllowList(getConnectedAllowList(node).value);
            return allLoras.filter(
                (name) =>
                    isInFolder(name, folderWidget.value) &&
                    isAllowed(name, folderWidget.value, allowed),
            );
        },
        set(values) {
            if (Array.isArray(values)) {
                allLoras = [...values];
                syncSelection();
            }
        },
    };

    function syncSelection() {
        const filtered = valuesDescriptor.get();
        if (!filtered.includes(loraWidget.value)) {
            loraWidget.value = filtered[0] ?? "";
        }
        node.setDirtyCanvas?.(true, true);
    }

    Object.defineProperty(loraWidget.options, "values", valuesDescriptor);

    const originalCallback = folderWidget.callback;
    folderWidget.callback = function (value) {
        const result = originalCallback?.apply(this, arguments);
        syncSelection();
        return result;
    };

    syncSelection();
    return syncSelection;
}

function updateBankVisibility(node) {
    const bankCount = getBankCount(node);
    for (let bankIndex = 2; bankIndex <= MAX_BANKS; bankIndex += 1) {
        const hidden = bankIndex > bankCount;
        for (const name of BANK_WIDGET_NAMES) {
            const widget = getWidget(node, getBankWidgetName(name, bankIndex));
            if (widget) {
                widget.hidden = hidden;
            }
        }
    }

    const size = node.computeSize?.();
    if (size) {
        node.setSize?.(size);
    }
    node.setDirtyCanvas?.(true, true);
}

function configureFilter(node) {
    const folderWidget = getWidget(node, "folder");
    if (!folderWidget) {
        return;
    }

    const syncSelections = [];
    for (let bankIndex = 1; bankIndex <= MAX_BANKS; bankIndex += 1) {
        const loraWidget = getWidget(node, getBankWidgetName("lora_name", bankIndex));
        if (!loraWidget) {
            continue;
        }
        const syncSelection = configureLoraFilter(node, folderWidget, loraWidget);
        if (syncSelection) {
            syncSelections.push(syncSelection);
        }
    }
    if (syncSelections.length === 0) {
        return;
    }

    function sync() {
        const { sourceWidget } = getConnectedAllowList(node);
        if (node.enviralAllowListSource && node.enviralAllowListSource !== sourceWidget) {
            node.enviralAllowListSource.enviralLoraFilterTargets?.delete(node);
        }
        node.enviralAllowListSource = sourceWidget;
        if (sourceWidget) {
            sourceWidget.enviralLoraFilterTargets ??= new Set();
            sourceWidget.enviralLoraFilterTargets.add(node);
            if (!sourceWidget.enviralLoraFilterWrapped) {
                const originalCallback = sourceWidget.callback;
                sourceWidget.callback = function (value) {
                    const result = originalCallback?.apply(this, arguments);
                    for (const target of sourceWidget.enviralLoraFilterTargets) {
                        target.enviralSyncLoraFilter?.();
                    }
                    return result;
                };
                sourceWidget.enviralLoraFilterWrapped = true;
            }
        }
        for (const syncSelection of syncSelections) {
            syncSelection();
        }
        updateBankVisibility(node);
    }

    const bankCountWidget = getWidget(node, "bank_count");
    if (bankCountWidget) {
        const originalCallback = bankCountWidget.callback;
        bankCountWidget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            sync();
            return result;
        };
    }

    const originalCallback = folderWidget.callback;
    folderWidget.callback = function (value) {
        const result = originalCallback?.apply(this, arguments);
        sync();
        return result;
    };

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        queueMicrotask(() => this.enviralSyncLoraFilter?.());
        return result;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        this.enviralAllowListSource?.enviralLoraFilterTargets?.delete(this);
        return onRemoved?.apply(this, arguments);
    };

    node.enviralSyncLoraFilter = sync;
    sync();
}

app.registerExtension({
    name: "EnviralDesign.FilteredLoraLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            configureFilter(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            queueMicrotask(() => this.enviralSyncLoraFilter?.());
            return result;
        };
    },
});
