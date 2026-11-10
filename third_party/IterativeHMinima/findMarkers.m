function [allMarkers, allMarkersOrg] = findMarkers(map, binMask, tArea, dSize, initH)

    [dx, dy] = size(map);

    allMarkers = zeros(dx, dy);
    allMarkersOrg = zeros(dx, dy);

    h = initH;

    while 1

        if max(map(:)) <= 0
            return;
        end

        % Step 1: H-minima transform
        hmap = imhmin(map, h);

        % Step 2: Find regional minima
        currMarkers = imregionalmin(hmap);
        [~, nRaw] = bwlabel(currMarkers);

        % Step 3: Area filtering
        currMarkers = maskBasedOnArea(currMarkers, tArea);
        [~, nArea] = bwlabel(currMarkers);

        % Step 4: Foreground mask filtering
        currMarkers = maskBasedOnMap(currMarkers, binMask);
        [~, nMask] = bwlabel(currMarkers);

        % Step 5: Overlap filtering
        currMarkers = eliminateOverlappingAreas( ...
            allMarkers, currMarkers, dx, dy);
        [~, nOverlap] = bwlabel(currMarkers);

        % Diagnostic output
        fprintf('H=%g | Initial=%d | Area=%d | Mask=%d | Overlap=%d\n', ...
            h, nRaw, nArea, nMask, nOverlap);

        % Step 6: Insert accepted markers
        [allMarkers, allMarkersOrg, inserted] = ...
            insertNewMarkers(allMarkers, allMarkersOrg, currMarkers, dSize);

        if inserted == 0
            return;
        else
            h = h + 1;
        end

    end

end


function markers = maskBasedOnArea(markers, tArea)

    if tArea ~= -1
        markers = bwareaopen(markers, tArea);
    end

end


function markers = maskBasedOnMap(markers, binMask)

    [L, n] = bwlabel(markers);

    for i = 1:n

        t = (L == i);

        overlapRatio = sum(binMask(t)) / sum(t(:));

        if overlapRatio < 1
            markers(t) = 0;
        end

    end

end


function curr = eliminateOverlappingAreas(prev, curr, dx, dy)

    cc = bwlabeln(curr);

    maxCC = max(cc(:));

    overlappingCC = zeros(maxCC, 1);

    for i = 1:dx
        for j = 1:dy

            if cc(i,j) > 0

                if prev(i,j) > 0
                    overlappingCC(cc(i,j)) = ...
                        overlappingCC(cc(i,j)) + 1;
                end

            end

        end
    end

    eliminateCC = zeros(maxCC, 1);

    for i = 1:maxCC

        if overlappingCC(i) > 0
            eliminateCC(i) = 1;
        end

    end

    for i = 1:dx
        for j = 1:dy

            if cc(i,j) > 0 && eliminateCC(cc(i,j)) == 1
                curr(i,j) = 0;
            end

        end
    end

end


function [allMarkers, allMarkersOrg, inserted] = ...
    insertNewMarkers(allMarkers, allMarkersOrg, currMarkers, dSize)

    maxNo = max(allMarkers(:));

    cc = bwlabeln(currMarkers);

    if dSize ~= -1

        se = strel('disk', dSize);
        cc2 = imdilate(cc, se);

    else

        cc2 = cc;

    end

    inserted = max(cc(:));

    cc2(cc2 > 0) = cc2(cc2 > 0) + maxNo;

    allMarkers(cc2 > 0) = cc2(cc2 > 0);

    cc(cc > 0) = cc(cc > 0) + maxNo;

    allMarkersOrg(cc > 0) = cc(cc > 0);

end