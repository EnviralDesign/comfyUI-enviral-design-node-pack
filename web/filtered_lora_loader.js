import { app } from "/scripts/app.js";

const NODE_NAME = "EnviralLoadLoraFiltered";
const ALL_FOLDERS = "All LoRAs";

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

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function configureFilter(node) {
    const folderWidget = getWidget(node, "folder");
    const loraWidget = getWidget(node, "lora_name");
    if (!folderWidget || !loraWidget || !Array.isArray(loraWidget.options?.values)) {
        return;
    }

    let allLoras = [...loraWidget.options.values];
    const valuesDescriptor = {
        configurable: true,
        enumerable: true,
        get() {
            return allLoras.filter((name) => isInFolder(name, folderWidget.value));
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
    node.enviralSyncLoraFilter = syncSelection;
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
